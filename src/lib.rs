use wgpu::{CommandEncoder, ComputePassDescriptor, Device, TextureView};

pub struct DrawContext {
    device: Device,
}

pub struct DrawList {

}

impl DrawContext {
    pub fn new(device: Device) -> DrawContext {
        DrawContext {
            device
        }
    }

    pub fn render(&mut self, encoder: &mut CommandEncoder, list: &mut DrawList, texture: TextureView) {
        let compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Canvas Compute Pass"),
            timestamp_writes: None
        });
    }
}