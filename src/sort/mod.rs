use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages};

use crate::alloc::{Allocation, AllocationMemory};

pub(super) struct SortPipeline {
    count_keys: ComputePipeline,
    count_histograms: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    uniform_offsets: Vec<u32>,
    uniforms: Buffer,
    device: Device,
}

pub(super) struct SortBuffers {
    pub(super) keys: Allocation<u32>,
    pub(super) temp_keys: Allocation<u32>,
    pub(super) histograms: Allocation<[u32; 16]>
}

impl SortPipeline {
    pub(super) fn new(device: Device) -> SortPipeline {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Sort Shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into())
        });

        let bind_group_layout_entires = [0, 1, 2, 3].map(|i| {
            BindGroupLayoutEntry {
                binding: i,
                count: None,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: if i == 3 {
                        BufferBindingType::Uniform
                    } else {
                        BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: i == 3,
                    min_binding_size: None
                }
            }
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sort Bind Group Layout"),
            entries: bind_group_layout_entires.as_slice()
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Sort Pipeline Layout"),
            immediate_size: 0,
            bind_group_layouts: &[Some(&bind_group_layout)]
        });

        let count_keys = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Count Keys Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("countKeys"),
            layout: Some(&pipeline_layout),
            module: &module
        });

        let count_histograms = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Count Histograms Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("countHistograms"),
            layout: Some(&pipeline_layout),
            module: &module
        });

        let mut uniform_offsets = Vec::new();
        let mut uniform_contents = Vec::new();
        let alignment = device.limits().min_uniform_buffer_offset_alignment;

        for i in (0..u32::BITS).step_by(4) {
            let len = uniform_contents.len();
            let offset = len.next_multiple_of(alignment as usize);
            uniform_contents.resize(offset, 0);
            uniform_offsets.push(offset as u32);
            uniform_contents.extend_from_slice(&i.to_ne_bytes());
        }

        let uniforms = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &uniform_contents ,
            usage: BufferUsages::UNIFORM
        });

        SortPipeline {
            bind_group_layout,
            count_histograms,
            uniform_offsets,
            count_keys,
            uniforms,
            device
        }
    }

    pub(super) fn encode(&self, compute_pass: &mut ComputePass, memory: &mut AllocationMemory, buffers: SortBuffers) {
        let Some(keys) = memory.binding(buffers.keys) else { return };
        let Some(temp_keys) = memory.binding(buffers.temp_keys) else { return };
        let Some(histograms) = memory.binding(buffers.histograms) else { return };
        let uniforms = self.uniforms.as_entire_binding();

        let mut binding = 0;
        let bind_group_entries = [keys, temp_keys, histograms, uniforms].map(|resource| {
            let entry = BindGroupEntry { binding, resource };
            binding += 1;
            entry
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Sort Bind Group"),
            layout: &self.bind_group_layout,
            entries: bind_group_entries.as_slice()
        });

        for offset in self.uniform_offsets.iter().copied() {
            let x = buffers.keys.len().div_ceil(256) as u32;
            compute_pass.set_bind_group(0, &bind_group, &[offset]);
            compute_pass.set_pipeline(&self.count_keys);
            compute_pass.dispatch_workgroups(x, 1, 1);
            compute_pass.set_pipeline(&self.count_histograms);

            //Only for testing purposes
            break
        }
    }
}