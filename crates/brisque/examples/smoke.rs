use brisque::Brisque;
use za::zenpixels::{PixelView, formats};
use za::{EvalOpts, Image, NoReferenceMetric};
use zenmetrics_abstractions as za;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let img = image::open(&path).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let slice = PixelView::new(&img, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    match Brisque::new().evaluate_image(Image::from(slice), EvalOpts::new()) {
        Ok(e) => println!("{:.9}", e.score()),
        Err(e) => println!("ERR {:?}", e.error().kind()),
    }
}
