@group(0) @binding(0)
var<storage, read_write> points: array<Point>;
@group(0) @binding(1)
var<storage, read_write> paths: array<Path>;

@group(0) @binding(2)
var<storage, read_write> triangleListIndices: array<u32>;
@group(0) @binding(3)
var<storage, read_write> triangles: array<Triangle>;

@group(0) @binding(4)
var<storage, read_write> uniforms: Params;

struct Params {
    sampleFractionBits: u32
}

struct Point {
    value: vec2i,
    path: u32
}

struct Path {
    fractionBits: u32,
    material: u32,
    offset: u32,
    length: u32
}

struct Triangle {
    clockwise: u32,
    material: u32,
    a: vec2u,
    b: vec2u,
    c: vec2u
}

struct Builtin {
    @builtin(global_invocation_id) global_index: vec3u
}

@compute @workgroup_size(256)
fn main(builtin: Builtin) {
    let index = builtin.global_index.x;
    if index >= arrayLength(&points) { return; }
    let point = points[index];
    let path = paths[point.path];

    let localTriangleIndex = index - path.offset;
    let pointCount = path.length;

    if (localTriangleIndex + 2u >= pointCount) {
        //degenerate triangle
    }

    let survivor = 1u << firstLeadingBit(pointCount - 1u);
    var indexA = localTriangleIndex + 1u;
    indexA += u32(indexA >= survivor);

    let stride = 1u << firstTrailingBit(indexA);
    let indexB = indexA - stride;
    var indexC = indexA + stride;
    indexC = select(indexC, 0u, indexC >= pointCount);

    let vertexA = points[indexA].value;
    let vertexB = points[indexB].value;
    let vertexC = points[indexC].value;

    let fractionBits = path.fractionBits;
    let edgeA = clipLine(array(vertexA, vertexB), fractionBits);
    let edgeB = clipLine(array(vertexB, vertexC), fractionBits);
    let edgeC = clipLine(array(vertexC, vertexA), fractionBits);
    let flip = isClockwise(vertexA, vertexB, vertexC);

    var triangle: Triangle;
    triangle.clockwise = u32(flip);
    triangle.material = path.material;
    triangle.a = makeEdge(edgeA[0], edgeA[1], flip);
    triangle.b = makeEdge(edgeB[0], edgeB[1], flip);
    triangle.c = makeEdge(edgeC[0], edgeC[1], flip);
    triangles[index] = triangle;
}

fn isClockwise(a: vec2i, b: vec2i, c: vec2i) -> bool {
    var au = bitcast<vec2u>(a) ^ BIAS;
    var bu = bitcast<vec2u>(b) ^ BIAS;
    var cu = bitcast<vec2u>(c) ^ BIAS;

    let delta_ba_mag = max(bu, au) - min(bu, au);
    let delta_ca_mag = max(cu, au) - min(cu, au);
    let delta_ba_zero = delta_ba_mag == vec2(0u);
    let delta_ca_zero = delta_ca_mag == vec2(0u);
    let zero = delta_ba_zero | delta_ca_zero.yx;
    let delta_ba_neg = au > bu;
    let delta_ca_neg = au > cu;

    let p = mulU32x2(delta_ba_mag, delta_ca_mag.yx);
    let n = (delta_ba_neg != delta_ca_neg.yx) & !zero;
    let m = select(p[0].x > p[0].y, p[1].x > p[1].y, p[0].x == p[0].y);
    let l = select(p[0].x < p[0].y, p[1].x < p[1].y, p[0].x == p[0].y);
    return !select(select(m, l, n.x), n.y, n.x != n.y); 
}

fn makeEdge(p0: vec2u, p1: vec2u, flip: bool) -> vec2u {
    let tp0 = select(vec2i(p0), vec2i(p1), flip);
    let tp1 = select(vec2i(p1), vec2i(p0), flip);

    let a = tp0.y - tp1.y;
    let b = tp1.x - tp0.x;
    var c = tp0.x * tp1.y - tp1.x * tp0.y;

    let d = tp1 - tp0;
    let topLeft = (d.y < 0i) || (d.y == 0i && d.x > 0i);
    c -= i32(!topLeft);

    var packedAB = bitcast<u32>(a) & MASK;
    packedAB |= (bitcast<u32>(b) & MASK) << SHIFT;
    return vec2u(packedAB, bitcast<u32>(c));
}

const BIAS = vec2(0x80000000u);

fn clipLine(points: array<vec2i, 2>, fractionBits: u32) -> array<vec2u, 2> {
    let sample = uniforms.sampleFractionBits;
    let world = fractionBits;
    let sample_shift = max(sample, world) - world; 
    let input_shift = max(sample, world) - sample;

    var p0 = bitcast<vec2u>(roundShiftRightI32(points[0], input_shift)) ^ BIAS;
    var p1 = bitcast<vec2u>(roundShiftRightI32(points[1], input_shift)) ^ BIAS;
    let degen = all(p0 == p1);

    let mirror = p1 < p0;
    p0 = select(p0, ~p0, mirror);
    p1 = select(p1, ~p1, mirror); 

    var uclip_min = vec2(0u) + BIAS;
    var uclip_max = vec2(0x7fffu >> sample_shift) + BIAS;
    let clip_min = select(uclip_min, ~uclip_max, mirror);
    let clip_max = select(uclip_max, ~uclip_min, mirror);

    let line = array(p0, p1);
    let c = clipUnsignedLine(line, clip_min, clip_max, sample_shift);
    var c0 = c[0];
    var c1 = c[1];

    c0 = select(c0, ~c0, mirror) ^ BIAS;
    c1 = select(c1, ~c1, mirror) ^ BIAS;
    c0 = select(c0, vec2(0u), degen);
    c1 = select(c1, vec2(0u), degen);
    return array(c0, c1);
}

fn roundShiftRightI32(p: vec2i, shift: u32) -> vec2i {
    let s = vec2u(shift);
    let half = bitcast<vec2i>(vec2i(1i) << vec2u(shift - 1u));
    let bias = select(half, -half, p < vec2(0i));
    return select(bitcast<vec2i>((p + bias) >> s), p, shift == 0u);
}

fn clipUnsignedLine(points: array<vec2u, 2>, clip_min: vec2u, clip_max: vec2u, sample_shift: u32) -> array<vec2u, 2> {
    var p0 = points[0];
    var p1 = points[1];
    let delta = p1 - p0;

    let top_left = vec2(clip_min.x, clip_max.y);
    let bottom_right = vec2(clip_max.x, clip_min.y);

    let tlm = mulU32x2(delta.yx, absDiffU32x2(p0, top_left));
    let brm = mulU32x2(delta.yx, absDiffU32x2(p0, bottom_right));
    let blm = mulU32x2(delta.yx, absDiffU32x2(p0, clip_min));
    let trm = mulU32x2(delta.yx, absDiffU32x2(p0, clip_max));

    let above = rightOfLine(p0, delta, top_left, tlm);
    let below = leftOfLine(p0, delta, bottom_right, brm);
    let enters_left = rightOfLine(p0, delta, clip_min, blm);
    let exits_right = leftOfLine(p0, delta, clip_max, trm);

    let entry_mag = crossMagnitude(p0, clip_min, blm);
    let exit_mag  = crossMagnitude(p0, clip_max, trm);
    let entry_den = select(delta.y, delta.x, enters_left);
    let exit_den = select(delta.y, delta.x, exits_right);

    p0 = scaleClipCoord(clip_min, sample_shift);
    p1 = scaleClipCoord(clip_max, sample_shift);
    p0[u32(enters_left)] += divU64byU32ToU15(entry_mag, entry_den, sample_shift);
    p1[u32(exits_right)] -= divU64byU32ToU15(exit_mag, exit_den, sample_shift);

    let fallback = select(bottom_right, top_left, enters_left);
    let invalid = above || below || all(p0 == p1);
    p0 = select(p0, clip_min, invalid);
    p1 = select(p1, fallback, invalid);
    return array(p0, p1);
}

fn scaleClipCoord(p: vec2u, sample_shift: u32) -> vec2u {
    let mirrored = p < BIAS;
    let shift = vec2(sample_shift);
    var unbiased = select(p, ~p, mirrored) ^ BIAS;
    let result = (unbiased << shift) ^ BIAS;
    return select(result, ~result, mirrored);
}

const SHIFT = 16u;
const MASK = 0xffffu;
const SHIFT2 = vec2(SHIFT);
const MASK2 = vec2(MASK);

fn crossMagnitude(origin: vec2u, p: vec2u, m: array<vec2u, 2>) -> vec2u {
    let hi = m[0];
    let lo = m[1];
    let opposite_signs = (p.x < origin.x) != (p.y < origin.y);

    let sum_lo = lo.x + lo.y;
    let sum_carry = u32(sum_lo < lo.x);
    let sum_hi = hi.x + hi.y + sum_carry;

    let x_ge_y = (hi.x > hi.y) || ((hi.x == hi.y) && (lo.x >= lo.y));

    let big_hi = select(hi.y, hi.x, x_ge_y);
    let big_lo = select(lo.y, lo.x, x_ge_y);
    let small_hi = select(hi.x, hi.y, x_ge_y);
    let small_lo = select(lo.x, lo.y, x_ge_y);

    let diff_lo = big_lo - small_lo;
    let borrow = u32(big_lo < small_lo);
    let diff_hi = big_hi - small_hi - borrow;

    return select(
        vec2u(diff_hi, diff_lo),
        vec2u(sum_hi, sum_lo),
        opposite_signs,
    );
}

fn divU64byU32ToU15(n: vec2u, d: u32, fractionBits: u32) -> u32 {
    var m = shl64Bit0To31(n, fractionBits);
    let denominator = max(1u, d);

    let half = denominator >> 1u;
    let lo = m.y + half;
    let carry = select(0u, 1u, lo < m.y);
    m = vec2u(m.x + carry, lo);

    let norm_shift = countLeadingZeros(denominator);
    let v = denominator << norm_shift;
    let u = shl64Bit0To31(m, norm_shift);

    let u2 = u.x & MASK;
    let u1 = u.y >> SHIFT;
    let u0 = u.y & MASK;

    let v1 = v >> SHIFT;
    let v0 = v & MASK;

    let top = (u2 << SHIFT) | u1;

    let q = top / v1;
    let r = top - q * v1;
    let too_big = q * v0 > ((r << SHIFT) | u0);
    return q - select(0u, 1u, too_big);
}

fn mulU32x2(a: vec2u, b: vec2u) -> array<vec2u, 2> {
    let al = a & MASK2;
    let ah = a >> SHIFT2;
    let bl = b & MASK2;
    let bh = b >> SHIFT2;

    let p0 = al * bl;
    let p1 = ah * bl + (p0 >> SHIFT2);
    let p2 = al * bh + (p1 & MASK2);

    let hi = ah * bh + (p1 >> SHIFT2) + (p2 >> SHIFT2);
    let lo = (p2 << SHIFT2) | (p0 & MASK2);
    return array(hi, lo); 
}

fn shl64Bit0To31(v: vec2u, s: u32) -> vec2u {
    let ss = s & 31u;
    let carry_shift = (32u - ss) & 31u;
    let carry_mask = select(0u, 0xffffffffu, ss != 0u);
    return vec2u(
        (v.x << ss) | ((v.y >> carry_shift) & carry_mask),
        v.y << ss
    );
}

fn rightOfLine(origin: vec2u, delta: vec2u, p: vec2u, m: array<vec2u, 2>) -> bool {
    var neg = p < origin;
    var hi = m[0];
    var lo = m[1];

    neg &= (hi | lo) != vec2(0u);
    hi = select(hi, hi.yx, neg.x);
    lo = select(lo, lo.yx, neg.x);
    
    let gt = (hi.x > hi.y) || ((hi.x == hi.y) && (lo.x > lo.y));
    return select(neg.y, gt, neg.x == neg.y);
}

fn leftOfLine(origin: vec2u, delta: vec2u, p: vec2u, m: array<vec2u, 2>) -> bool {
    return rightOfLine(origin.yx, delta.yx, p.yx, array(m[0].yx, m[1].yx));
}

fn absDiffU32x2(a: vec2u, b: vec2u) -> vec2u {
    return max(a, b) - min(a, b);
}