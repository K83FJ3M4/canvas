@group(0) @binding(0)
var<storage, read_write> keysA: array<u32>;

@group(0) @binding(1)
var<storage, read_write> keysB: array<u32>;

@group(0) @binding(2)
var<storage, read_write> histograms: array<array<u32, 16>>;

@group(0) @binding(3)
var<uniform> shift: u32;

var<workgroup> localKeys: array<u32, 256>;
var<workgroup> offsetHistogram: array<u32, 16>;
var<workgroup> countHistograms: array<array<u32, 16>, 16>;

struct Builtin {
    @builtin(global_invocation_id) globalIndex: vec3u,
    @builtin(local_invocation_id) localIndex: vec3u,
    @builtin(workgroup_id) workgroupIndex: vec3u,
    @builtin(num_workgroups) numWorkgroups: vec3u,
}

@compute @workgroup_size(256)
fn scanKeys(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.x;
    if (localIndex < 16u) { offsetHistogram[localIndex] = 0u; }

    let key = loadLocalKeys(globalIndex);
    workgroupBarrier();

    countTileKeys(globalIndex);
    workgroupBarrier(); 

    scanLocalHistograms(localIndex);
    workgroupBarrier();

    scatterKey(globalIndex, key);
}

@compute @workgroup_size(256)
fn scanHistograms(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.x;
    let numWorkgroups = builtin.numWorkgroups.x;
    let workgroupIndex = builtin.workgroupIndex.x;
    let histogramRange = getHistogramRange(numWorkgroups);

    loadOffsetHistogram(globalIndex, histogramRange); 

    let lane = localIndex & 15u;
    let tile = localIndex >> 4;

    var count = 0u;
    let globalOffset = histogramRange.x;
    let workgroupOffset = globalOffset + workgroupIndex * 256u;
    let tileStart = workgroupOffset + tile * 16u;
    let tileEnd = min(tileStart + 16u, histogramRange.y);

    for (var i = tileStart; i < tileEnd; i++) {
        count += histograms[i][lane];
    }

    countHistograms[tile][lane] = count;
    workgroupBarrier();

    scanLocalHistograms(localIndex);
    workgroupBarrier();
    count = countHistograms[tile][lane];

    for (var i = tileStart; i < tileEnd; i++) {
        let value = histograms[i][lane];
        histograms[i][lane] = count;
        count += value;
    }
}

@compute @workgroup_size(16)
fn initOffsetHistogram(builtin: Builtin) {
    let histogramRange = getHistogramRange(1);
    let localIndex = builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.x;

    loadOffsetHistogram(globalIndex, histogramRange);
    workgroupBarrier();

    var sum = 0u;
    if (localIndex == 0u) {
        for (var i = 0u; i < 16u; i++) {
            let count = offsetHistogram[i];
            offsetHistogram[i] = sum;
            sum += count;
        }
    }

    workgroupBarrier();
    histograms[histogramRange.y][localIndex] = offsetHistogram[localIndex];
}

@compute @workgroup_size(256)
fn countKeys(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.x;
    let workgroupIndex = builtin.workgroupIndex.x;

    loadLocalKeys(globalIndex);
    workgroupBarrier();

    countTileKeys(globalIndex);
    workgroupBarrier();

    if (localIndex >= 16u) { return; }

    var count = 0u;
    for (var i = 0u; i < 16u; i++) {
        count += countHistograms[i][localIndex];
    }
    histograms[workgroupIndex][localIndex] = count;
}

@compute @workgroup_size(256)
fn mergeHistograms(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let numWorkgroups = builtin.numWorkgroups.x;
    let workgroupIndex = builtin.workgroupIndex.x;
    let histogramRange = getHistogramRange(numWorkgroups);

    let lane = localIndex & 15u;
    let tile = localIndex >> 4;

    var count = 0u;
    let globalOffset = histogramRange.x;
    let workgroupOffset = globalOffset + workgroupIndex * 256u;
    let tileStart = workgroupOffset + tile * 16u;
    let tileEnd = min(tileStart + 16u, histogramRange.y);

    for (var i = tileStart; i < tileEnd; i++) {
        count += histograms[i][lane];
    }

    countHistograms[tile][lane] = count;
    workgroupBarrier();

    if (localIndex >= 16u) { return; }
    offsetHistogram[localIndex] = 0u;
    let sum = scanLocalHistograms(localIndex);
    let targetIndex = histogramRange.y + workgroupIndex;
    histograms[targetIndex][localIndex] = sum;
}

fn getHistogramRange(numWorkgroups: u32) -> vec2u {
    var offset = 0u;
    var sourceCount = divCeil256(keyArrayLength());
    var outputCount = divCeil256(sourceCount);
    while (numWorkgroups < outputCount) {
        offset += sourceCount;
        sourceCount = outputCount;
        outputCount = divCeil256(sourceCount);
    }

    return vec2(offset, offset + sourceCount);
}

fn scanLocalHistograms(localIndex: u32) -> u32 {
    if (localIndex >= 16u) { return 0u; }

    var sum = offsetHistogram[localIndex];
    for (var i = 0u; i < 16u; i++) {
        let value = countHistograms[i][localIndex];
        countHistograms[i][localIndex] = sum;
        sum += value;
    }

    return sum;
}

fn scatterKey(globalIndex: u32, key: u32) {
    if globalIndex >= keyArrayLength() { return; }
    let workgroupIndex = globalIndex >> 8u;
    let localIndex = globalIndex & 255u;
    let tile = localIndex >> 4u;
    let digit = (key >> shift) & 15u;
    var index = countHistograms[tile][digit];

    for (var i = tile * 16u; i < localIndex; i++) {
        var otherDigit = (localKeys[i] >> shift) & 15u;
        index += u32(otherDigit == digit);
    }

    index += histograms[workgroupIndex][digit];

    if (bool(shift & 4u)) {
        keysA[index] = key;
    } else {
        keysB[index] = key;
    }
}

fn countTileKeys(globalIndex: u32) {
    let workgroupOffset = globalIndex & ~255u;
    let numKeys = keyArrayLength() - workgroupOffset;
    let localIndex = globalIndex & 255u;
    let lane = localIndex & 15u;
    let tile = localIndex >> 4u;
    let tileOffset = tile * 16u;

    var count = 0u;
    for (var i = tileOffset; i < tileOffset + 16u; i++) {
        let digit = (localKeys[i] >> shift) & 15u;
        count += u32(digit == lane && i < numKeys);
    }

    countHistograms[tile][lane] = count;
}

fn loadLocalKeys(globalIndex: u32) -> u32 {
    let validA = globalIndex < arrayLength(&keysA);
    let validB = globalIndex < arrayLength(&keysB);
    let localIndex = globalIndex & 255u;
    let sourceB = bool(shift & 4u);
    var key = 0u;

    if (validA && !sourceB) {
        key = keysA[globalIndex];
    } else if (validB && sourceB) {
        key = keysB[globalIndex];
    }

    localKeys[localIndex] = key;
    return key;
}

fn loadOffsetHistogram(globalIndex: u32, histogramRange: vec2u) {
    let localIndex = globalIndex & 255u;
    let workgroupIndex = globalIndex >> 8;
    if (localIndex >= 16u) { return; }
    let index = histogramRange.y + workgroupIndex;
    offsetHistogram[localIndex] = histograms[index][localIndex];
}

fn keyArrayLength() -> u32 {
    if (bool(shift & 4u)) {
        return arrayLength(&keysB);
    } else {
        return arrayLength(&keysA);
    }
}

fn divCeil256(num: u32) -> u32 {
    return ((num - 1u) >> 8u) + 1u;
}