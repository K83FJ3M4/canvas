use std::iter::{once, successors};

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType, BufferSize, BufferUsages, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages};

use crate::alloc::{Allocation, AllocationMemory};

pub(super) struct SortPipeline {
    count_keys: ComputePipeline,
    merge_histograms: ComputePipeline,
    init_offset_histogram: ComputePipeline,
    scan_histograms: ComputePipeline,
    scan_keys: ComputePipeline,
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

        let merge_histograms = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Merge Histograms Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("mergeHistograms"),
            layout: Some(&pipeline_layout),
            module: &module
        });

        let init_offset_histogram = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Init Offset Histogram Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("initOffsetHistogram"),
            layout: Some(&pipeline_layout),
            module: &module
        });

        let scan_histograms = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Scan Histograms Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("scanHistograms"),
            layout: Some(&pipeline_layout),
            module: &module
        });

        let scan_keys = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Scan Keys Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("scanKeys"),
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
            init_offset_histogram,
            bind_group_layout,
            merge_histograms,
            scan_histograms,
            uniform_offsets,
            count_keys,
            scan_keys,
            uniforms,
            device
        }
    }

    pub(super) fn encode(&self, compute_pass: &mut ComputePass, memory: &mut AllocationMemory, buffers: SortBuffers) {
        let Some(keys) = memory.binding(buffers.keys) else { return };
        let Some(temp_keys) = memory.binding(buffers.temp_keys) else { return };
        let Some(histograms) = memory.binding(buffers.histograms) else { return };
        let uniforms = BindingResource::Buffer(BufferBinding {
            buffer: &self.uniforms,
            size: BufferSize::new(4),
            offset: 0,
        });

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

        let length = buffers.keys.len() as u32;
        let reduction = |item: &u32| Some(item.div_ceil(256));
        let mut count_iter = successors(Some(length), reduction)
            .take_while(|count| *count > 1).skip(1);
        let init_dispatch_size = count_iter.next().unwrap_or(1);
        let dispatch_sizes = count_iter.chain(once(1))
            .collect::<Vec<u32>>();

        for offset in self.uniform_offsets.iter().copied() { 
            compute_pass.set_bind_group(0, &bind_group, &[offset]);
            compute_pass.set_pipeline(&self.count_keys);
            compute_pass.dispatch_workgroups(init_dispatch_size, 1, 1);
            compute_pass.set_pipeline(&self.merge_histograms);

            for dispatch_size in dispatch_sizes.iter().copied() {
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            }

            compute_pass.set_pipeline(&self.init_offset_histogram);
            compute_pass.dispatch_workgroups(1, 1, 1);
            compute_pass.set_pipeline(&self.scan_histograms);

            for dispatch_size in dispatch_sizes.iter().copied().rev() {
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            }

            compute_pass.set_pipeline(&self.scan_keys);
            compute_pass.dispatch_workgroups(init_dispatch_size, 1, 1);
        }
    }
}