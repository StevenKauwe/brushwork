//! brushwork's canvas page (https://stevenkauwe.github.io/brushwork/): lay down flat shapes, the way a film feeds
//! the engine, then paint them in oil. The page ([`app`]) runs the UI; a Web Worker ([`worker`]) runs the CPU
//! engine so the page never freezes. [`scene`] is the shapes and the film they become. Built by `tools/build-web.sh`.

pub mod app;
pub mod scene;
pub mod worker;
