//! The GPU backend: the same strokes, laid by the graphics card.
//!
//! The CPU still plans every stroke (`Painting::stroke` works out everything that is the same for the whole
//! stroke: its colours after premix and fade, the pigment-mixing table, what the brush picked up from the plain
//! picture, its mask). With a [`Recorder`] on the painting, those finished strokes are written down instead of laid.
//! At the end of the frame they are binned, in order, into 16 x 16 tiles, and one GPU thread per pixel walks its
//! tile's strokes in order, doing for its own pixel exactly what the CPU loop does. A pixel only ever reads and
//! writes itself, and the plain picture the brushes pick up from stays on the CPU (it is captured into each stroke
//! when it is planned), so the order per pixel, and therefore the painting, is the CPU's.
//!
//! What can differ: the GPU's `sin`, `exp`, `pow` and division are not bit-for-bit the CPU library's, so a pixel can
//! differ in the last bits, and very rarely a pixel right at a threshold (the edge of a bristle, the end of a stroke)
//! falls the other way. The CPU engine stays the reference; the tests run on the GPU with `OIL_TEST_BACKEND=gpu`.
//!
//! Waiting for the GPU: opening the device and reading the canvas back are `async` (`open`, `render_async`), with one
//! copy of the work for both platforms. Natively the device is polled until the work is done, so the future is ready
//! the first time it is polled and `render` is just `pollster::block_on` of it. In a browser nothing may block (the
//! mapping only completes when the browser's event loop runs), so the canvas page awaits it in its Web Worker; the
//! browser's own WebGPU is then the driver, and a lost device or rejected work comes back as an error (the page
//! paints that frame on the CPU) instead of a panic.

// Without the `gpu` feature nothing records strokes (the backend can't be chosen), so the recorder's fields go unread.
#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

/// One planned stroke (or, with `kind == 1`, a full-frame glaze), as the GPU reads it. All scalars, 4 bytes each.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Cmd {
    pub cx: f32,
    pub cy: f32,
    pub sn: f32,
    pub cs: f32,
    pub len: f32,
    pub width: f32,
    pub bend: f32,
    pub along: f32,
    pub span0: f32,
    pub span1: f32,
    pub opacity: f32,
    pub pickup: f32,
    pub load: f32,
    pub col: [f32; 3],
    pub alt: [f32; 3],
    pub under: [f32; 3],
    pub nb: f32,
    pub marks: f32,
    pub lips: f32,
    pub edge_soft: f32,
    pub whole_len: f32,
    pub end_w: f32,
    pub slant_k: f32,
    pub slant_c: f32,
    pub ridge_k: f32,
    pub size_k: f32,
    pub knife_k: f32,
    pub z: f32,
    pub half_max: f32,
    pub id: u32,
    /// 0 flat, 1 filbert, 2 round, 3 knife
    pub tool: u32,
    pub pigment: u32,
    /// index of the first of this stroke's 18 pigment-table colours (x 3 floats)
    pub worked: u32,
    /// mask: offset into the mask buffer, origin, size (w = 0: no mask)
    pub mask_off: u32,
    pub mask_ox: i32,
    pub mask_oy: i32,
    pub mask_w: u32,
    pub mask_h: u32,
    pub x0: i32,
    pub x1: i32,
    pub y0: i32,
    pub y1: i32,
    /// 0 stroke, 1 glaze (col = the glaze colour, opacity = its strength)
    pub kind: u32,
    pub _pad: [u32; 15],
}
// SAFETY: Cmd is repr(C), made only of f32/u32/i32 (and arrays of them), with no padding (64 x 4 bytes).
unsafe impl bytemuck::Zeroable for Cmd {}
unsafe impl bytemuck::Pod for Cmd {}
const _: () = assert!(std::mem::size_of::<Cmd>() == 64 * 4);

/// The strokes of one frame, in the order they were planned, and what they need.
pub(crate) struct Recorder {
    pub cmds: Vec<Cmd>,
    /// Masks, losslessly in 16 x 16 blocks: per mask a table with one entry per block (0: all 0.0, 1: all 1.0,
    /// n >= 2: the block's 256 values start at index n - 2 of this same buffer, as f32 bits). Big shapes are
    /// almost all inside or outside, so this is a fraction of the bytes of the plain mask.
    pub masks: Vec<u32>,
    pub worked: Vec<f32>,
    /// the mask most recently handed over by `use_mask`: its offset, origin and size
    pub mask: Option<(u32, [i32; 2], [usize; 2])>,
    /// per stroke: the tile rows it can touch, with the x range in each (for binning)
    pub bands: Vec<Vec<(u32, i32, i32)>>,
    /// the canvas before the first stroke (the toned ground)
    pub ground: Vec<[f32; 3]>,
}

impl Recorder {
    pub fn new(ground: Vec<[f32; 3]>) -> Recorder {
        Recorder { cmds: vec![], masks: vec![], worked: vec![], mask: None, bands: vec![], ground }
    }
    /// The mask the next strokes are clipped to (a shape's, from `paint_form`).
    pub fn use_mask(&mut self, m: &[f32], o: [i32; 2], sz: [usize; 2]) {
        let (bw, bh) = (sz[0].div_ceil(TILE), sz[1].div_ceil(TILE));
        let off = self.masks.len();
        self.masks.resize(off + bw * bh, 0);
        for by in 0..bh {
            for bx in 0..bw {
                let at = |x: usize, y: usize| -> f32 {
                    let (px, py) = (bx * TILE + x, by * TILE + y);
                    if px < sz[0] && py < sz[1] { m[py * sz[0] + px] } else { 0.0 }
                };
                let (mut zero, mut one) = (true, true);
                for y in 0..TILE {
                    for x in 0..TILE {
                        let v = at(x, y);
                        zero &= v.to_bits() == 0f32.to_bits();
                        one &= v.to_bits() == 1f32.to_bits();
                    }
                }
                self.masks[off + by * bw + bx] = if zero {
                    0
                } else if one {
                    1
                } else {
                    let data = self.masks.len() as u32;
                    for y in 0..TILE {
                        for x in 0..TILE {
                            self.masks.push(at(x, y).to_bits());
                        }
                    }
                    data + 2
                };
            }
        }
        self.mask = Some((off as u32, o, sz));
    }
    pub fn push(&mut self, mut c: Cmd, worked: Option<&[[[f32; 3]; 9]; 2]>, bands: Vec<(u32, i32, i32)>) {
        if let Some(w) = worked {
            c.worked = (self.worked.len() / 3) as u32;
            for side in w {
                for rgb in side {
                    self.worked.extend_from_slice(rgb);
                }
            }
        }
        self.cmds.push(c);
        self.bands.push(bands);
    }
    pub fn glaze(&mut self, col: [f32; 3], k: f32) {
        let c = Cmd { col, opacity: k, kind: 1, x0: 0, x1: super::WI as i32 - 1, y0: 0, y1: super::HI as i32 - 1, ..Default::default() };
        let bands = (0..super::HI.div_ceil(TILE) as u32).map(|t| (t, 0, super::WI as i32 - 1)).collect();
        self.cmds.push(c);
        self.bands.push(bands);
    }
}

pub(crate) const TILE: usize = 16;

/// Bin the strokes into tiles, in order: per tile, the indices of the strokes that can touch it.
#[cfg(feature = "gpu")]
pub(crate) fn bin(r: &Recorder) -> (Vec<u32>, Vec<u32>) {
    let (tw, th) = (super::WI.div_ceil(TILE), super::HI.div_ceil(TILE));
    let mut lists: Vec<Vec<u32>> = vec![vec![]; tw * th];
    for (i, bands) in r.bands.iter().enumerate() {
        for &(ty, lo, hi) in bands {
            for tx in (lo.max(0) as usize / TILE)..=((hi.max(0) as usize).min(super::WI - 1) / TILE) {
                lists[ty as usize * tw + tx].push(i as u32);
            }
        }
    }
    let mut offsets = Vec::with_capacity(lists.len() + 1);
    let mut entries = Vec::new();
    offsets.push(0u32);
    for l in &lists {
        entries.extend_from_slice(l);
        offsets.push(entries.len() as u32);
    }
    if entries.is_empty() {
        entries.push(0);
    }
    (offsets, entries)
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub(crate) use backend::render;
#[cfg(feature = "gpu")]
pub(crate) use backend::{open, render_async};

#[cfg(feature = "gpu")]
mod backend {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};
    use wgpu::util::DeviceExt;

    struct Gpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        /// the toned ground, uploaded once (every GPU frame starts from the same one), with a hash to check it
        ground: Mutex<Option<(u64, Arc<wgpu::Buffer>)>>,
        /// set by wgpu when the device goes away (a driver reset, the browser reclaiming it): the next `open`
        /// makes a new one
        lost: Arc<AtomicBool>,
        /// what the adapter calls itself (often empty in a browser, which keeps it private)
        name: String,
    }

    /// The device every frame uses, made by the first `open` (natively also by the first frame).
    static GPU: Mutex<Option<Arc<Gpu>>> = Mutex::new(None);

    impl Gpu {
        async fn new() -> Result<Gpu, String> {
            #[cfg(not(target_arch = "wasm32"))]
            let desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
            #[cfg(target_arch = "wasm32")]
            let desc = wgpu::InstanceDescriptor { backends: wgpu::Backends::BROWSER_WEBGPU, ..wgpu::InstanceDescriptor::new_without_display_handle() };
            let instance = wgpu::Instance::new(desc);
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
                .await
                .map_err(|e| if cfg!(target_arch = "wasm32") { "the browser offers no WebGPU adapter".to_string() } else { format!("no GPU adapter ({e})") })?;
            let info = adapter.get_info();
            // A browser's software stand-in (SwiftShader and the like) is slower than the CPU engine in its worker.
            if cfg!(target_arch = "wasm32") && info.device_type == wgpu::DeviceType::Cpu {
                return Err("the browser offers only a software GPU".into());
            }
            let limits = adapter.limits();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { label: Some("oil"), required_limits: limits, ..Default::default() })
                .await
                .map_err(|e| format!("no GPU device ({e})"))?;
            let lost = Arc::new(AtomicBool::new(false));
            let flag = lost.clone();
            device.set_device_lost_callback(move |_, _| flag.store(true, Ordering::Relaxed));
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oil strokes"),
                source: wgpu::ShaderSource::Wgsl(include_str!("strokes.wgsl").into()),
            });
            let entry = |b: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("oil"),
                entries: &[entry(0, true), entry(1, true), entry(2, true), entry(3, true), entry(4, true), entry(5, false), entry(6, false)],
            });
            let pl =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("oil"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oil strokes"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok(Gpu { device, queue, pipeline, layout, ground: Mutex::new(None), lost, name: info.name })
        }
    }

    fn current() -> Option<Arc<Gpu>> {
        GPU.lock().unwrap_or_else(|e| e.into_inner()).clone().filter(|g| !g.lost.load(Ordering::Relaxed))
    }

    /// Make the GPU device ready (again, if the last one was lost): what the adapter calls itself, or why there is
    /// none. In a browser this is the only way to get one, as nothing there may wait for the GPU without `await`.
    pub(crate) async fn open() -> Result<String, String> {
        if let Some(g) = current() {
            return Ok(g.name.clone());
        }
        let g = Arc::new(Gpu::new().await?);
        let name = g.name.clone();
        *GPU.lock().unwrap_or_else(|e| e.into_inner()) = Some(g);
        Ok(name)
    }

    /// A value a wgpu callback hands over later, as a future (no executor of its own: whoever awaits drives it).
    struct Later<T>(Arc<Mutex<(Option<T>, Option<Waker>)>>);
    fn later<T: Send + 'static>() -> (impl FnOnce(T) + Send + 'static, Later<T>) {
        let slot = Arc::new(Mutex::new((None, None::<Waker>)));
        let tx = slot.clone();
        let send = move |v: T| {
            let mut s = tx.lock().unwrap_or_else(|e| e.into_inner());
            s.0 = Some(v);
            if let Some(w) = s.1.take() {
                w.wake();
            }
        };
        (send, Later(slot))
    }
    impl<T> Future for Later<T> {
        type Output = T;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
            let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
            match s.0.take() {
                Some(v) => Poll::Ready(v),
                None => {
                    s.1 = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }

    /// Lay the recorded strokes on the GPU, blocking until they are read back (native only): the canvas colour
    /// and paint thickness. The same work as [`render_async`], which a browser has to use.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn render(r: &Recorder) -> (Vec<[f32; 3]>, Vec<f32>) {
        pollster::block_on(render_async(r)).unwrap_or_else(|e| panic!("oil painter: {e}"))
    }

    /// Lay the recorded strokes on the GPU and read them back: the canvas colour and paint thickness. Nothing in
    /// here blocks in a browser: the readback completes when the browser resolves the mapping and wakes the
    /// awaiting task. Natively the device is polled until the work is done, so the future is ready when first
    /// polled. An error if the device is lost or rejects the work.
    pub(crate) async fn render_async(r: &Recorder) -> Result<(Vec<[f32; 3]>, Vec<f32>), String> {
        #[cfg(not(target_arch = "wasm32"))]
        if current().is_none() {
            open().await?;
        }
        let g = current().ok_or("the GPU has not been opened, or was lost")?;
        // (no clock unless asked for: there is none on wasm32-unknown-unknown)
        let t0 = super::super::timing().then(std::time::Instant::now);
        let since = |t: Option<std::time::Instant>| t.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let (offsets, entries) = bin(r);
        let t_bin = since(t0);
        let n = super::super::WI * super::super::HI;
        // the ground: the same for every frame, so it lives on the GPU and is copied into each frame's canvas
        let key = r.ground.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, c| c.iter().fold(h, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3)));
        let ground = {
            let mut g0 = g.ground.lock().unwrap_or_else(|e| e.into_inner());
            match g0.as_ref() {
                Some((k, b)) if *k == key => b.clone(),
                _ => {
                    let col0: Vec<f32> = r.ground.iter().flat_map(|c| c.iter().copied()).collect();
                    let b = Arc::new(g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("ground"),
                        contents: bytemuck::cast_slice(&col0),
                        usage: wgpu::BufferUsages::COPY_SRC,
                    }));
                    *g0 = Some((key, b.clone()));
                    b
                }
            }
        };
        let scope_mem = g.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope_valid = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let nonempty = |v: &[f32]| if v.is_empty() { vec![0.0f32] } else { v.to_vec() };
        let cmds: &[Cmd] = if r.cmds.is_empty() { &[Cmd { kind: 2, ..Default::default() }] } else { &r.cmds };
        let mk = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
        };
        let ro = wgpu::BufferUsages::STORAGE;
        let b_cmds = mk("cmds", bytemuck::cast_slice(cmds), ro);
        let masks: &[u32] = if r.masks.is_empty() { &[0] } else { &r.masks };
        let b_masks = mk("masks", bytemuck::cast_slice(masks), ro);
        let b_worked = mk("worked", bytemuck::cast_slice(&nonempty(&r.worked)), ro);
        let b_off = mk("offsets", bytemuck::cast_slice(&offsets), ro);
        let b_ent = mk("entries", bytemuck::cast_slice(&entries), ro);
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        // (wgpu hands out new buffers zeroed: the thickness starts at nothing without an upload)
        let b_col = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("col"), size: (n * 12) as u64, usage: rw, mapped_at_creation: false });
        let b_thick = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("thick"), size: (n * 4) as u64, usage: rw, mapped_at_creation: false });
        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oil"),
            layout: &g.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_cmds.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_masks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_worked.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_off.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_ent.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_col.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b_thick.as_entire_binding() },
            ],
        });
        let read = |label: &str, size: u64| {
            g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let r_col = read("col read", (n * 12) as u64);
        let r_thick = read("thick read", (n * 4) as u64);
        let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("oil") });
        enc.copy_buffer_to_buffer(&ground, 0, &b_col, 0, (n * 12) as u64);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("strokes"), timestamp_writes: None });
            pass.set_pipeline(&g.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(super::super::WI.div_ceil(TILE) as u32, super::super::HI.div_ceil(TILE) as u32, 1);
        }
        enc.copy_buffer_to_buffer(&b_col, 0, &r_col, 0, (n * 12) as u64);
        enc.copy_buffer_to_buffer(&b_thick, 0, &r_thick, 0, (n * 4) as u64);
        let t_up = since(t0);
        g.queue.submit([enc.finish()]);
        let (sc, st) = (r_col.slice(..), r_thick.slice(..));
        let (tx_col, rx_col) = later();
        let (tx_thick, rx_thick) = later();
        sc.map_async(wgpu::MapMode::Read, tx_col);
        st.map_async(wgpu::MapMode::Read, tx_thick);
        // Natively the work is driven here and the mapping callbacks fire inside this call; a browser does that
        // itself, between turns of its event loop, while the awaits below hand control back to it.
        #[cfg(not(target_arch = "wasm32"))]
        g.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll: {e}"))?;
        let (valid, mem) = (scope_valid.pop(), scope_mem.pop());
        let (valid, mem) = (valid.await, mem.await);
        if let Some(e) = valid.or(mem) {
            return Err(format!("the GPU rejected the strokes: {e}"));
        }
        rx_col.await.map_err(|e| format!("reading the colour back: {e}"))?;
        rx_thick.await.map_err(|e| format!("reading the thickness back: {e}"))?;
        let t_gpu = since(t0);
        let col: Vec<[f32; 3]> =
            bytemuck::cast_slice::<u8, f32>(&sc.get_mapped_range().map_err(|e| format!("mapping the colour: {e}"))?).as_chunks::<3>().0.to_vec();
        let thick: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&st.get_mapped_range().map_err(|e| format!("mapping the thickness: {e}"))?).to_vec();
        if t0.is_some() {
            let t_all = since(t0);
            eprintln!(
                "TIMING render bin {:.3} upload {:.3} gpu+wait {:.3} readback {:.3} ({} tile entries)",
                t_bin,
                t_up - t_bin,
                t_gpu - t_up,
                t_all - t_gpu,
                entries.len()
            );
        }
        Ok((col, thick))
    }
}
