use std::borrow::Cow;

use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat, TextureView, TextureViewDimension};

use crate::alloc::{Allocation, AllocationMemory};

pub(super) struct TilePipeline {
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    device: Device
}

pub(super) struct TileBuffers {
    pub(super) params: Allocation<u32>,
    pub(super) texture: TextureView,
    pub(super) lists: Allocation<u32>,
    pub(super) list_ranges: Allocation<[u32; 2]>
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

        let bind_group_layout_entries = [0, 1, 2, 3].map(|i| BindGroupLayoutEntry {
            binding: i,
            visibility: ShaderStages::COMPUTE,
            count: None,
            ty: if i == 0 {
                BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    view_dimension: TextureViewDimension::D2,
                    format
                }
            } else {
                BindingType::Buffer {
                    has_dynamic_offset: false,
                    min_binding_size: None,
                    ty: BufferBindingType::Storage { read_only: false }
                }
            }
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Tile Bind Group Layout"),
            entries: &bind_group_layout_entries
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

    pub(super) fn encode(&self, compute_pass: &mut ComputePass, memory: &mut AllocationMemory, buffers: TileBuffers) {
        let Some(params) = memory.binding(buffers.params) else { return };
        let Some(lists) = memory.binding(buffers.lists) else { return };
        let Some(list_ranges) = memory.binding(buffers.list_ranges) else { return };
        let texture = BindingResource::TextureView(&buffers.texture);

        let mut binding = 0;
        let bind_group_entries = [texture, lists, list_ranges, params].map(|resource| {
            let entry = BindGroupEntry { binding, resource };
            binding += 1;
            entry
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Tile Bind Group"),
            layout: &self.bind_group_layout,
            entries: &bind_group_entries
        });

        let x = buffers.texture.texture().width().div_ceil(16);
        let y = buffers.texture.texture().height().div_ceil(16);
        compute_pass.set_bind_group(0, &bind_group, Default::default());
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.dispatch_workgroups(x, y, 1);
    }
}