use std::borrow::Cow;

use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat, TextureView, TextureViewDimension};

pub(super) struct TilePipeline {
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    device: Device
}

impl TilePipeline {
    pub(super) fn new(device: Device, format: TextureFormat) -> TilePipeline {
        let mut source = match format {
            TextureFormat::Rgba8Unorm => include_str!("output_rgba.wgsl").to_string(),
            TextureFormat::Bgra8Unorm => include_str!("output_bgra.wgsl").to_string(),
            format => panic!("Canvas format must be either Rgba8Unorm or Bgra8Unorm not {format:?}")
        };

        source.push_str(include_str!("shader.wgsl"));
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Tile Shader"),
            source: ShaderSource::Wgsl(Cow::Owned(source))
        });

        let texture_layout_entry = BindGroupLayoutEntry {
            binding: 0,
            count: None,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                view_dimension: TextureViewDimension::D2,
                format
            }
        };

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Tile Bind Group Layout"),
            entries: &[texture_layout_entry]
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Tile Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Tile Pipeline"),
            module: &module,
            entry_point: Some("main"),
            layout: Some(&pipeline_layout),
            compilation_options: Default::default(),
            cache: None
        });

        TilePipeline {
            compute_pipeline,
            bind_group_layout,
            device
        }
    }

    pub(super) fn encode(&self, compute_pass: &mut ComputePass, texture: TextureView) {
        let texture_entry = BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&texture)
        };

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Tile Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[texture_entry]
        });

        let x = texture.texture().width().div_ceil(16);
        let y = texture.texture().height().div_ceil(16);
        compute_pass.set_bind_group(0, &bind_group, Default::default());
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.dispatch_workgroups(x, y, 1);
    }
}