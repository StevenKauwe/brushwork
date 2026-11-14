//! Browser binding for the fast native/WebGPU photo painter. Call in a Web Worker.
use brushwork::{Dynamics, FastPhotoOptions, FastPhotoResolution, PhotoOptions};
use wasm_bindgen::prelude::*;

/// Returns {pixels, width, height, backend, detail, milliseconds}. Alpha is preserved.
/// Fast WebGPU is attempted unless force_cpu; errors explicitly fall back to Detailed CPU.
#[wasm_bindgen]
pub async fn paint_photo_rgba(rgba: Vec<u8>, width: u32, height: u32, resolution: u32, seed: u32, steady: bool, force_cpu: bool) -> Result<JsValue, JsValue> {
    let err = |s: &str| JsValue::from_str(s);
    if width == 0 || height == 0 || width > 4096 || height > 4096 || rgba.len() != width as usize * height as usize * 4 {
        return Err(err("Expected RGBA input, 1–4096 pixels per side"));
    }
    let resolution = match resolution {
        270 => FastPhotoResolution::P270,
        540 => FastPhotoResolution::P540,
        720 => FastPhotoResolution::P720,
        1080 => FastPhotoResolution::P1080,
        _ => return Err(err("Choose 270, 540, 720 or 1080")),
    };
    let source = image::RgbImage::from_fn(width, height, |x, y| {
        let i = (y * width + x) as usize * 4;
        image::Rgb([rgba[i], rgba[i + 1], rgba[i + 2]])
    });
    let options = PhotoOptions { attention: false, ground_subject: true, ..Default::default() };
    let started = js_sys::Date::now();
    let (mut backend, mut detail) = ("cpu", String::from("Detailed CPU requested"));
    let fast = if force_cpu {
        None
    } else {
        match brushwork::open_gpu().await {
            Ok(adapter) => match brushwork::paint_photo_image_fast_async(
                source.clone(),
                &options,
                [0.5; 3],
                Dynamics::default(),
                seed,
                FastPhotoOptions { resolution, steady, upscale: true },
                &mut |_, _, _| Ok(()),
            )
            .await
            {
                Ok(image) => {
                    backend = "gpu";
                    detail = if adapter.is_empty() { "WebGPU".into() } else { adapter };
                    Some(image)
                }
                Err(e) => {
                    detail = format!("WebGPU failed: {e}; used Detailed CPU");
                    None
                }
            },
            Err(e) => {
                detail = format!("WebGPU unavailable: {e}; used Detailed CPU");
                None
            }
        }
    };
    let painted = match fast {
        Some(image) => image,
        None => {
            brushwork::set_backend(brushwork::Backend::Cpu);
            brushwork::paint_photo_image(source, &options, [0.5; 3], Dynamics::default(), seed, &mut |_, _, _| {}).map_err(|s| err(&s))?
        }
    };
    let painted = brushwork::resize_photo_fast(&painted, width, height);
    let mut pixels = Vec::with_capacity(rgba.len());
    for (rgb, alpha) in painted.as_raw().as_chunks::<3>().0.iter().zip(rgba.iter().skip(3).step_by(4)) {
        pixels.extend_from_slice(rgb);
        pixels.push(*alpha);
    }
    let result = js_sys::Object::new();
    for (key, value) in [
        ("pixels", js_sys::Uint8ClampedArray::from(pixels.as_slice()).into()),
        ("width", JsValue::from(width)),
        ("height", JsValue::from(height)),
        ("backend", JsValue::from_str(backend)),
        ("detail", JsValue::from_str(&detail)),
        ("milliseconds", JsValue::from_f64(js_sys::Date::now() - started)),
    ] {
        js_sys::Reflect::set(&result, &JsValue::from_str(key), &value)?;
    }
    Ok(result.into())
}
