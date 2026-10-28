//! webcheck: opens the canvas page in a real browser and checks that a painting appears.
//!
//! ```text
//! tools/build-web.sh && cargo run -p webcheck -- [--dir dist | --url URL] [--browser webkit|chrome] [--phone] [--shot out.png]
//! ```
//!
//! It serves `--dir` (default `dist`) on a local port, or opens `--url`, through WebDriver: `WebKitWebDriver` with
//! WebKitGTK's MiniBrowser (the engine of every browser on iPhone; CI installs `webkit2gtk-driver` and runs under
//! `xvfb-run`), or `chromedriver` with Chromium. `--phone` opens it at an iPhone's size. It waits for the sample scene
//! to finish painting (the canvas's `data-state` turns `painted`), checks the canvas holds a picture (many colours,
//! not one), prints the paint time from `data-paint-ms`, and with `--shot` saves a screenshot.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine;
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, String>;

const IPHONE_UA: &str =
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Mobile/15E148 Safari/604.1";

/// A plain HTTP/1.1 request to localhost; returns the body. Drivers don't always close the connection, so the
/// body is read by its `Content-Length` (or its chunks), not to the end of the stream.
fn http(port: u16, method: &str, path: &str, body: &str) -> Result<String> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(Duration::from_secs(300))).map_err(|e| e.to_string())?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut buf: Vec<u8> = vec![];
    let mut chunk = [0u8; 65536];
    let read_more = |s: &mut TcpStream, buf: &mut Vec<u8>, chunk: &mut [u8]| -> Result<bool> {
        let n = s.read(chunk).map_err(|e| e.to_string())?;
        buf.extend_from_slice(&chunk[..n]);
        Ok(n > 0)
    };
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if !read_more(&mut s, &mut buf, &mut chunk)? {
            return Err("connection closed before the headers".into());
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
    let header = |name: &str| head.lines().find_map(|l| l.strip_prefix(name).map(|v| v.trim().to_string()));
    if let Some(len) = header("content-length:").and_then(|v| v.parse::<usize>().ok()) {
        while buf.len() < head_end + len {
            if !read_more(&mut s, &mut buf, &mut chunk)? {
                break;
            }
        }
        return Ok(String::from_utf8_lossy(&buf[head_end..(head_end + len).min(buf.len())]).into_owned());
    }
    if header("transfer-encoding:").is_some_and(|v| v.contains("chunked")) {
        // chunks until the empty one
        let mut out = vec![];
        let mut at = head_end;
        loop {
            let line_end = loop {
                if let Some(i) = buf[at..].windows(2).position(|w| w == b"\r\n") {
                    break at + i;
                }
                if !read_more(&mut s, &mut buf, &mut chunk)? {
                    return Ok(String::from_utf8_lossy(&out).into_owned());
                }
            };
            let n = usize::from_str_radix(String::from_utf8_lossy(&buf[at..line_end]).trim(), 16).map_err(|e| e.to_string())?;
            if n == 0 {
                return Ok(String::from_utf8_lossy(&out).into_owned());
            }
            while buf.len() < line_end + 2 + n + 2 {
                if !read_more(&mut s, &mut buf, &mut chunk)? {
                    break;
                }
            }
            out.extend_from_slice(&buf[line_end + 2..(line_end + 2 + n).min(buf.len())]);
            at = line_end + 2 + n + 2;
        }
    }
    while read_more(&mut s, &mut buf, &mut chunk)? {}
    Ok(String::from_utf8_lossy(&buf[head_end..]).into_owned())
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?.local_addr().map_err(|e| e.to_string())?.port())
}

fn which(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(|n| {
        let p = Path::new(n);
        if p.is_absolute() {
            return p.exists().then(|| p.to_path_buf());
        }
        std::env::var_os("PATH")?.to_str()?.split(':').map(|d| Path::new(d).join(n)).find(|p| p.exists())
    })
}

/// Serve a folder over HTTP on a local port, in the background (for as long as the program runs).
fn serve(dir: PathBuf) -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let path = req.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap_or("/").to_string();
            let rel = if path.ends_with('/') { format!("{path}index.html") } else { path };
            let file = dir.join(rel.trim_start_matches('/'));
            let (status, body) = match (file.starts_with(&dir) && !rel.contains(".."), std::fs::read(&file)) {
                (true, Ok(b)) => ("200 OK", b),
                _ => ("404 Not Found", b"not found".to_vec()),
            };
            let mime = match file.extension().and_then(|e| e.to_str()) {
                Some("html") => "text/html; charset=utf-8",
                Some("js") => "text/javascript",
                Some("wasm") => "application/wasm",
                Some("png") => "image/png",
                _ => "application/octet-stream",
            };
            let head = format!("HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s.write_all(head.as_bytes()).and_then(|_| s.write_all(&body));
        }
    });
    Ok(port)
}

struct Browser {
    port: u16,
    session: String,
    child: Child,
}

impl Browser {
    fn start(kind: &str, phone: bool) -> Result<Browser> {
        let port = free_port()?;
        let (driver, caps) = match kind {
            "webkit" => {
                let driver = which(&["WebKitWebDriver"]).ok_or("WebKitWebDriver not found (Ubuntu: apt install webkit2gtk-driver)")?;
                let mini = which(&[
                    "/usr/lib/x86_64-linux-gnu/webkit2gtk-4.1/MiniBrowser",
                    "/usr/lib/webkit2gtk-4.1/MiniBrowser",
                    "/usr/libexec/webkit2gtk-4.1/MiniBrowser",
                ])
                .ok_or("MiniBrowser (WebKitGTK 4.1) not found")?;
                let mut args = vec!["--automation".to_string()];
                if phone {
                    args.push(format!("--user-agent={IPHONE_UA}"));
                }
                (driver, json!({ "capabilities": { "alwaysMatch": { "webkitgtk:browserOptions": { "binary": mini, "args": args } } } }))
            }
            _ => {
                let driver = which(&["chromedriver"]).ok_or("chromedriver not found")?;
                let bin = which(&["chromium", "chromium-browser", "google-chrome"]).ok_or("Chromium not found")?;
                let size = if phone { "--window-size=390,844" } else { "--window-size=1280,900" };
                let mut args = vec!["--headless=new".to_string(), size.to_string(), "--no-sandbox".to_string()];
                if phone {
                    args.push(format!("--user-agent={IPHONE_UA}"));
                }
                (driver, json!({ "capabilities": { "alwaysMatch": { "browserName": "chrome", "goog:chromeOptions": { "binary": bin, "args": args } } } }))
            }
        };
        let child = Command::new(&driver)
            .arg(format!("--port={port}"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("{}: {e}", driver.display()))?;
        let t0 = Instant::now();
        while http(port, "GET", "/status", "").is_err() {
            if t0.elapsed() > Duration::from_secs(20) {
                return Err("the WebDriver didn't start".into());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let mut b = Browser { port, session: String::new(), child };
        let v = b.call("POST", "/session", Some(caps))?;
        b.session = format!("/session/{}", v["sessionId"].as_str().ok_or_else(|| format!("no session: {v}"))?);
        if phone && kind == "webkit" {
            let _ = b.call("POST", "/window/rect", Some(json!({ "width": 390, "height": 844 })));
        }
        Ok(b)
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value> {
        let body = body.map(|b| b.to_string()).unwrap_or_else(|| if method == "POST" { "{}".into() } else { String::new() });
        let text = http(self.port, method, &format!("{}{path}", self.session), &body)?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{method} {path}: {e}: {text}"))?;
        let v = v.get("value").cloned().unwrap_or(Value::Null);
        if let Some(err) = v.get("error") {
            return Err(format!("{method} {path}: {err}: {}", v["message"]));
        }
        Ok(v)
    }

    fn js(&self, script: &str) -> Result<Value> {
        self.call("POST", "/execute/sync", Some(json!({ "script": script, "args": [] })))
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.call("DELETE", "", None);
        let _ = self.child.kill();
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let phone = args.iter().any(|a| a == "--phone");
    let kind = opt("--browser").unwrap_or_else(|| if which(&["WebKitWebDriver"]).is_some() { "webkit".into() } else { "chrome".into() });
    let url = match opt("--url") {
        Some(u) => u,
        None => {
            let dir = std::fs::canonicalize(opt("--dir").unwrap_or_else(|| "dist".into())).map_err(|e| format!("dist: {e} (run tools/build-web.sh)"))?;
            format!("http://127.0.0.1:{}/", serve(dir)?)
        }
    };
    let b = Browser::start(&kind, phone)?;
    b.call("POST", "/url", Some(json!({ "url": url })))?;
    let t0 = Instant::now();
    let mut last = String::new();
    loop {
        let state = b.js(
            r#"const c = document.getElementById("canvas"); return c ? c.dataset.state + " | " + document.getElementById("status").textContent : "no canvas";"#,
        )?;
        let state = state.as_str().unwrap_or("").to_string();
        if state != last {
            println!("[{:5.1} s] {state}", t0.elapsed().as_secs_f64());
            last = state.clone();
        }
        if state.starts_with("painted") {
            break;
        }
        if state.contains("failed") || state.contains("Could not") {
            return Err(format!("the page reports: {state}"));
        }
        if t0.elapsed() > Duration::from_secs(240) {
            return Err(format!("no painting after 4 minutes (last: {state})"));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    // a painting has many colours; the flat shapes it started from have only a dozen
    let colours = b.js(r#"const c = document.getElementById("canvas"); const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
           const seen = new Set(); for (let i = 0; i < d.length; i += 4 * 7) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
           return [seen.size, Number(c.dataset.paintMs || 0), c.width, c.height, innerWidth, document.documentElement.scrollWidth];"#)?;
    let n = colours[0].as_u64().unwrap_or(0);
    let (ms, w, h, inner, scroll) = (
        colours[1].as_f64().unwrap_or(0.0),
        colours[2].as_u64().unwrap_or(0),
        colours[3].as_u64().unwrap_or(0),
        colours[4].as_u64().unwrap_or(0),
        colours[5].as_u64().unwrap_or(0),
    );
    println!(
        "{kind}{}: painted {w}x{h} in {:.1} s, {n} distinct colours sampled; page {scroll} px wide in a {inner} px window",
        if phone { " (phone)" } else { "" },
        ms / 1000.0
    );
    if n < 500 {
        return Err(format!("the canvas has only {n} colours: not a painting"));
    }
    if scroll > inner + 1 {
        return Err(format!("the page scrolls sideways ({scroll} px in a {inner} px window)"));
    }
    // draw a shape: press on the canvas, go round a circle, let go; then Paint, and wait for the new painting
    let canvas = b.call("POST", "/element", Some(json!({ "using": "css selector", "value": "#canvas" })))?;
    let id = canvas.as_object().and_then(|o| o.values().next()).and_then(|v| v.as_str()).ok_or("no canvas element")?.to_string();
    let rect = b.js(r#"const r = document.getElementById("canvas").getBoundingClientRect(); return [r.width, r.height];"#)?;
    let (cw, ch) = (rect[0].as_f64().unwrap_or(300.0), rect[1].as_f64().unwrap_or(170.0));
    let mut moves =
        vec![json!({ "type": "pointerMove", "duration": 0, "origin": { "element-6066-11e4-a52e-4f735466cecf": id }, "x": (cw * 0.25) as i64, "y": 0 })];
    moves.push(json!({ "type": "pointerDown", "button": 0 }));
    for k in 1..=24 {
        let a = k as f64 / 24.0 * std::f64::consts::TAU;
        moves.push(json!({ "type": "pointerMove", "duration": 10, "origin": { "element-6066-11e4-a52e-4f735466cecf": id }, "x": (cw * 0.25 * a.cos()) as i64, "y": (ch * 0.25 * a.sin()) as i64 }));
    }
    moves.push(json!({ "type": "pointerUp", "button": 0 }));
    b.call(
        "POST",
        "/actions",
        Some(json!({ "actions": [{ "type": "pointer", "id": "finger", "parameters": { "pointerType": "mouse" }, "actions": moves }] })),
    )?;
    let drawn = b.js(r#"return document.getElementById("canvas").dataset.state;"#)?;
    if drawn.as_str() != Some("flat") {
        return Err(format!("drawing a shape didn't put the page back to its shapes (state {drawn})"));
    }
    b.js(r#"document.getElementById("paint").click(); return true;"#)?;
    let t1 = Instant::now();
    loop {
        let state = b.js(r#"return document.getElementById("canvas").dataset.state;"#)?;
        if state.as_str() == Some("painted") {
            break;
        }
        if t1.elapsed() > Duration::from_secs(240) {
            return Err(format!("the drawn shape never got painted (state {state})"));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    println!("drew a shape and painted it again in {:.1} s", t1.elapsed().as_secs_f64());
    if let Some(path) = opt("--shot") {
        let png = b.call("GET", "/screenshot", None)?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(png.as_str().unwrap_or("")).map_err(|e| e.to_string())?;
        std::fs::write(&path, bytes).map_err(|e| format!("{path}: {e}"))?;
        println!("screenshot: {path}");
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("webcheck: {e}");
        std::process::exit(1);
    }
}
