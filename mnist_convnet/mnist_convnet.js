
const mnist_convnet = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_7_7_8_4_4_4_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_25088:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_784:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_288:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 7 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var precast0 = gidx0;
  var precast1 = gidx1;
  var precast2 = lidx0;
  var alu0 = (gidx1*112);
  var alu1 = (lidx0*36);
  var val0 = data2_288[alu1];
  var val1 = data2_288[(alu1+1)];
  var val2 = data2_288[(alu1+2)];
  var val3 = data2_288[(alu1+3)];
  var val4 = data2_288[(alu1+4)];
  var val5 = data2_288[(alu1+5)];
  var val6 = data2_288[(alu1+6)];
  var val7 = data2_288[(alu1+7)];
  var val8 = data2_288[(alu1+8)];
  var val9 = data2_288[(alu1+9)];
  var val10 = data2_288[(alu1+10)];
  var val11 = data2_288[(alu1+11)];
  var val12 = data2_288[(alu1+12)];
  var val13 = data2_288[(alu1+13)];
  var val14 = data2_288[(alu1+14)];
  var val15 = data2_288[(alu1+15)];
  var val16 = data2_288[(alu1+16)];
  var val17 = data2_288[(alu1+17)];
  var val18 = data2_288[(alu1+18)];
  var val19 = data2_288[(alu1+19)];
  var val20 = data2_288[(alu1+20)];
  var val21 = data2_288[(alu1+21)];
  var val22 = data2_288[(alu1+22)];
  var val23 = data2_288[(alu1+23)];
  var val24 = data2_288[(alu1+24)];
  var val25 = data2_288[(alu1+25)];
  var val26 = data2_288[(alu1+26)];
  var val27 = data2_288[(alu1+27)];
  var val28 = data2_288[(alu1+28)];
  var val29 = data2_288[(alu1+29)];
  var val30 = data2_288[(alu1+30)];
  var val31 = data2_288[(alu1+31)];
  var val32 = data2_288[(alu1+32)];
  var val33 = data2_288[(alu1+33)];
  var val34 = data2_288[(alu1+34)];
  var val35 = data2_288[(alu1+35)];
  var alu2 = (lidx1*28);
  var precast3 = (bitcast<u32>(precast0)<<2u);
  var cast0 = bitcast<i32>(precast3);
  var alu3 = (alu0+alu2+cast0);
  var val36 = data1_784[alu3];
  var val37 = data1_784[(alu3+1)];
  var val38 = data1_784[(alu3+2)];
  var val39 = data1_784[(alu3+3)];
  var precast4 = (bitcast<u32>(precast1)<<2u);
  var precast5 = (bitcast<u32>(precast2)<<2u);
  var cast1 = bitcast<i32>(precast5);
  var val40 = data3_32[cast1];
  var val41 = data3_32[(cast1+1)];
  var val42 = data3_32[(cast1+2)];
  var val43 = data3_32[(cast1+3)];
  var alu4 = (gidx0<6);
  var val44 = select(0.0f, data1_784[(alu3+4)], alu4);
  var alu5 = ((lidx1+bitcast<i32>(precast4))<27);
  var val45 = select(0.0f, data1_784[(alu3+29)], alu5);
  var val46 = select(0.0f, data1_784[(alu3+30)], alu5);
  var val47 = select(0.0f, data1_784[(alu3+28)], alu5);
  var val48 = select(0.0f, data1_784[(alu3+31)], alu5);
  var alu6 = (0<gidx0);
  var val49 = select(0.0f, data1_784[(alu3+-1)], alu6);
  var alu7 = (0<(gidx1+lidx1));
  var val50 = select(0.0f, data1_784[(alu3+-28)], alu7);
  var val51 = select(0.0f, data1_784[(alu3+-27)], alu7);
  var val52 = select(0.0f, data1_784[(alu3+-26)], alu7);
  var val53 = select(0.0f, data1_784[(alu3+-25)], alu7);
  var val54 = select(0.0f, data1_784[(alu3+32)], (alu5&alu4));
  var val55 = select(0.0f, data1_784[(alu3+27)], (alu5&alu6));
  var val56 = select(0.0f, data1_784[(alu3+-24)], (alu7&alu4));
  var val57 = select(0.0f, data1_784[(alu3+-29)], (alu7&alu6));
  var alu8 = (cast0+alu0+(lidx0*3136)+alu2);
  var alu9 = ((val51*val0)+(val37*val3)+(val45*val6)+(val52*val1)+(val38*val4)+(val46*val7)+(val53*val2)+(val39*val5)+(val48*val8)+val40);
  var alu10 = ((val51*val9)+(val37*val12)+(val45*val15)+(val52*val10)+(val38*val13)+(val46*val16)+(val53*val11)+(val39*val14)+(val48*val17)+val41);
  var alu11 = ((val51*val18)+(val37*val21)+(val45*val24)+(val52*val19)+(val38*val22)+(val46*val25)+(val53*val20)+(val39*val23)+(val48*val26)+val42);
  var alu12 = ((val51*val27)+(val37*val30)+(val45*val33)+(val52*val28)+(val38*val31)+(val46*val34)+(val53*val29)+(val39*val32)+(val48*val35)+val43);
  var alu13 = ((val52*val0)+(val38*val3)+(val46*val6)+(val53*val1)+(val39*val4)+(val48*val7)+(val56*val2)+(val44*val5)+(val54*val8)+val40);
  var alu14 = ((val52*val9)+(val38*val12)+(val46*val15)+(val53*val10)+(val39*val13)+(val48*val16)+(val56*val11)+(val44*val14)+(val54*val17)+val41);
  var alu15 = ((val52*val18)+(val38*val21)+(val46*val24)+(val53*val19)+(val39*val22)+(val48*val25)+(val56*val20)+(val44*val23)+(val54*val26)+val42);
  var alu16 = ((val52*val27)+(val38*val30)+(val46*val33)+(val53*val28)+(val39*val31)+(val48*val34)+(val56*val29)+(val44*val32)+(val54*val35)+val43);
  var alu17 = ((val50*val0)+(val36*val3)+(val47*val6)+(val51*val1)+(val37*val4)+(val45*val7)+(val52*val2)+(val38*val5)+(val46*val8)+val40);
  var alu18 = ((val57*val0)+(val49*val3)+(val55*val6)+(val50*val1)+(val36*val4)+(val47*val7)+(val51*val2)+(val37*val5)+(val45*val8)+val40);
  var alu19 = ((val50*val9)+(val36*val12)+(val47*val15)+(val51*val10)+(val37*val13)+(val45*val16)+(val52*val11)+(val38*val14)+(val46*val17)+val41);
  var alu20 = ((val57*val9)+(val49*val12)+(val55*val15)+(val50*val10)+(val36*val13)+(val47*val16)+(val51*val11)+(val37*val14)+(val45*val17)+val41);
  var alu21 = ((val50*val18)+(val36*val21)+(val47*val24)+(val51*val19)+(val37*val22)+(val45*val25)+(val52*val20)+(val38*val23)+(val46*val26)+val42);
  var alu22 = ((val57*val18)+(val49*val21)+(val55*val24)+(val50*val19)+(val36*val22)+(val47*val25)+(val51*val20)+(val37*val23)+(val45*val26)+val42);
  var alu23 = ((val50*val27)+(val36*val30)+(val47*val33)+(val51*val28)+(val37*val31)+(val45*val34)+(val52*val29)+(val38*val32)+(val46*val35)+val43);
  var alu24 = ((val57*val27)+(val49*val30)+(val55*val33)+(val50*val28)+(val36*val31)+(val47*val34)+(val51*val29)+(val37*val32)+(val45*val35)+val43);
  var alu25 = select(0.0f,alu20,(0.0f<alu20));
  data0_25088[(alu8+784)] = alu25;
  var alu27 = select(0.0f,alu22,(0.0f<alu22));
  data0_25088[(alu8+1568)] = alu27;
  var alu29 = select(0.0f,alu24,(0.0f<alu24));
  data0_25088[(alu8+2352)] = alu29;
  var alu31 = select(0.0f,alu18,(0.0f<alu18));
  data0_25088[alu8] = alu31;
  var alu33 = select(0.0f,alu19,(0.0f<alu19));
  data0_25088[(alu8+785)] = alu33;
  var alu35 = select(0.0f,alu21,(0.0f<alu21));
  data0_25088[(alu8+1569)] = alu35;
  var alu37 = select(0.0f,alu23,(0.0f<alu23));
  data0_25088[(alu8+2353)] = alu37;
  var alu39 = select(0.0f,alu17,(0.0f<alu17));
  data0_25088[(alu8+1)] = alu39;
  var alu41 = select(0.0f,alu10,(0.0f<alu10));
  data0_25088[(alu8+786)] = alu41;
  var alu43 = select(0.0f,alu11,(0.0f<alu11));
  data0_25088[(alu8+1570)] = alu43;
  var alu45 = select(0.0f,alu12,(0.0f<alu12));
  data0_25088[(alu8+2354)] = alu45;
  var alu47 = select(0.0f,alu9,(0.0f<alu9));
  data0_25088[(alu8+2)] = alu47;
  var alu49 = select(0.0f,alu14,(0.0f<alu14));
  data0_25088[(alu8+787)] = alu49;
  var alu51 = select(0.0f,alu15,(0.0f<alu15));
  data0_25088[(alu8+1571)] = alu51;
  var alu53 = select(0.0f,alu16,(0.0f<alu16));
  data0_25088[(alu8+2355)] = alu53;
  var alu55 = select(0.0f,alu13,(0.0f<alu13));
  data0_25088[(alu8+3)] = alu55;
}`;

const r_7_7_16_4_4_4_32_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_50176:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_25088:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_18432:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_64:array<f32>;
@compute @workgroup_size(16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 7 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 4 */
  var alu0 = (lidx1*28);
  var alu1 = (gidx1*112);
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<2u);
  var cast0 = bitcast<i32>(precast1);
  var alu2 = (gidx0<6);
  var precast2 = gidx1;
  var precast3 = (bitcast<u32>(precast2)<<2u);
  var alu3 = ((lidx1+bitcast<i32>(precast3))<27);
  var alu4 = (0<(gidx1+lidx1));
  var alu5 = (0<gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var ridx1006 = 0; ridx1006 < 32; ridx1006++) {
    var alu22 = ((lidx0*1152)+(ridx1006*9));
    var val0 = data2_18432[alu22];
    var val1 = data2_18432[(alu22+1)];
    var val2 = data2_18432[(alu22+2)];
    var val3 = data2_18432[(alu22+3)];
    var val4 = data2_18432[(alu22+4)];
    var val5 = data2_18432[(alu22+5)];
    var val6 = data2_18432[(alu22+6)];
    var val7 = data2_18432[(alu22+7)];
    var val8 = data2_18432[(alu22+8)];
    var val9 = data2_18432[(alu22+288)];
    var val10 = data2_18432[(alu22+289)];
    var val11 = data2_18432[(alu22+290)];
    var val12 = data2_18432[(alu22+291)];
    var val13 = data2_18432[(alu22+292)];
    var val14 = data2_18432[(alu22+293)];
    var val15 = data2_18432[(alu22+294)];
    var val16 = data2_18432[(alu22+295)];
    var val17 = data2_18432[(alu22+296)];
    var val18 = data2_18432[(alu22+576)];
    var val19 = data2_18432[(alu22+577)];
    var val20 = data2_18432[(alu22+578)];
    var val21 = data2_18432[(alu22+579)];
    var val22 = data2_18432[(alu22+580)];
    var val23 = data2_18432[(alu22+581)];
    var val24 = data2_18432[(alu22+582)];
    var val25 = data2_18432[(alu22+583)];
    var val26 = data2_18432[(alu22+584)];
    var val27 = data2_18432[(alu22+864)];
    var val28 = data2_18432[(alu22+865)];
    var val29 = data2_18432[(alu22+866)];
    var val30 = data2_18432[(alu22+867)];
    var val31 = data2_18432[(alu22+868)];
    var val32 = data2_18432[(alu22+869)];
    var val33 = data2_18432[(alu22+870)];
    var val34 = data2_18432[(alu22+871)];
    var val35 = data2_18432[(alu22+872)];
    var alu23 = (alu1+alu0+(ridx1006*784)+cast0);
    var val36 = data1_25088[alu23];
    var val37 = select(0.0f, data1_25088[(alu23+-29)], (alu4&alu5));
    var val38 = select(0.0f, data1_25088[(alu23+-28)], alu4);
    var val39 = select(0.0f, data1_25088[(alu23+-27)], alu4);
    var val40 = select(0.0f, data1_25088[(alu23+-26)], alu4);
    var val41 = select(0.0f, data1_25088[(alu23+-25)], alu4);
    var val42 = select(0.0f, data1_25088[(alu23+-24)], (alu4&alu2));
    var val43 = select(0.0f, data1_25088[(alu23+-1)], alu5);
    var val44 = data1_25088[(alu23+1)];
    var val45 = data1_25088[(alu23+2)];
    var val46 = data1_25088[(alu23+3)];
    var val47 = select(0.0f, data1_25088[(alu23+4)], alu2);
    var val48 = select(0.0f, data1_25088[(alu23+27)], (alu3&alu5));
    var val49 = select(0.0f, data1_25088[(alu23+28)], alu3);
    var val50 = select(0.0f, data1_25088[(alu23+29)], alu3);
    var val51 = select(0.0f, data1_25088[(alu23+30)], alu3);
    var val52 = select(0.0f, data1_25088[(alu23+31)], alu3);
    var val53 = select(0.0f, data1_25088[(alu23+32)], (alu3&alu2));
    acc0[8] = (acc0[8]+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc0[9] = (acc0[9]+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc0[10] = (acc0[10]+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc0[11] = (acc0[11]+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc0[12] = (acc0[12]+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc0[13] = (acc0[13]+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc0[14] = (acc0[14]+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc0[15] = (acc0[15]+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
    acc0[4] = (acc0[4]+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc0[0] = (acc0[0]+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc0[5] = (acc0[5]+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc0[1] = (acc0[1]+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc0[6] = (acc0[6]+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc0[2] = (acc0[2]+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc0[7] = (acc0[7]+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc0[3] = (acc0[3]+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
  }
  var precast4 = lidx0;
  var precast5 = (bitcast<u32>(precast4)<<2u);
  var cast1 = bitcast<i32>(precast5);
  var val54 = data3_64[cast1];
  var val55 = data3_64[(cast1+1)];
  var val56 = data3_64[(cast1+2)];
  var val57 = data3_64[(cast1+3)];
  var alu41 = (cast0+alu1+(lidx0*3136)+alu0);
  data0_50176[alu41] = (acc0[0]+val54);
  data0_50176[(alu41+1)] = (acc0[4]+val54);
  data0_50176[(alu41+2)] = (acc0[8]+val54);
  data0_50176[(alu41+3)] = (acc0[12]+val54);
  data0_50176[(alu41+784)] = (acc0[1]+val55);
  data0_50176[(alu41+785)] = (acc0[5]+val55);
  data0_50176[(alu41+786)] = (acc0[9]+val55);
  data0_50176[(alu41+787)] = (acc0[13]+val55);
  data0_50176[(alu41+1568)] = (acc0[2]+val56);
  data0_50176[(alu41+1569)] = (acc0[6]+val56);
  data0_50176[(alu41+1570)] = (acc0[10]+val56);
  data0_50176[(alu41+1571)] = (acc0[14]+val56);
  data0_50176[(alu41+2352)] = (acc0[3]+val57);
  data0_50176[(alu41+2353)] = (acc0[7]+val57);
  data0_50176[(alu41+2354)] = (acc0[11]+val57);
  data0_50176[(alu41+2355)] = (acc0[15]+val57);
}`;

const r_7_7_32_2_4_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_12544:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_50176:array<f32>;
@compute @workgroup_size(32,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 7 */
  var lidx0 = i32(lindex.x); /* 32 */
  var lidx1 = i32(lindex.y); /* 2 */
  var precast0 = gidx0;
  var precast1 = lidx1;
  var cast0 = bitcast<u32>(precast0);
  var precast2 = (cast0<<2u);
  var precast3 = (bitcast<u32>(precast1)<<1u);
  var alu0 = (bitcast<i32>(precast2)+(gidx1*7168)+(lidx0*224)+bitcast<i32>(precast3));
  var val0 = data1_50176[alu0];
  var val1 = data1_50176[(alu0+1)];
  var val2 = data1_50176[(alu0+28)];
  var val3 = data1_50176[(alu0+29)];
  var val4 = data1_50176[(alu0+56)];
  var val5 = data1_50176[(alu0+57)];
  var val6 = data1_50176[(alu0+84)];
  var val7 = data1_50176[(alu0+85)];
  var val8 = data1_50176[(alu0+112)];
  var val9 = data1_50176[(alu0+113)];
  var val10 = data1_50176[(alu0+140)];
  var val11 = data1_50176[(alu0+141)];
  var val12 = data1_50176[(alu0+168)];
  var val13 = data1_50176[(alu0+169)];
  var val14 = data1_50176[(alu0+196)];
  var val15 = data1_50176[(alu0+197)];
  var precast4 = (cast0<<1u);
  var alu1 = (lidx1+bitcast<i32>(precast4)+(gidx1*1792)+(lidx0*56));
  var alu2 = select(0.0f,val0,(0.0f<val0));
  var alu3 = select(0.0f,val1,(0.0f<val1));
  var alu4 = select(0.0f,val2,(0.0f<val2));
  var alu5 = select(0.0f,val3,(0.0f<val3));
  data0_12544[alu1] = ((alu2+alu4+alu3+alu5)*0.25f);
  var alu7 = select(0.0f,val4,(0.0f<val4));
  var alu8 = select(0.0f,val5,(0.0f<val5));
  var alu9 = select(0.0f,val6,(0.0f<val6));
  var alu10 = select(0.0f,val7,(0.0f<val7));
  data0_12544[(alu1+14)] = ((alu7+alu9+alu8+alu10)*0.25f);
  var alu12 = select(0.0f,val8,(0.0f<val8));
  var alu13 = select(0.0f,val9,(0.0f<val9));
  var alu14 = select(0.0f,val10,(0.0f<val10));
  var alu15 = select(0.0f,val11,(0.0f<val11));
  data0_12544[(alu1+28)] = ((alu12+alu14+alu13+alu15)*0.25f);
  var alu17 = select(0.0f,val12,(0.0f<val12));
  var alu18 = select(0.0f,val13,(0.0f<val13));
  var alu19 = select(0.0f,val14,(0.0f<val14));
  var alu20 = select(0.0f,val15,(0.0f<val15));
  data0_12544[(alu1+42)] = ((alu17+alu19+alu18+alu20)*0.25f);
}`;

const r_7_7_32_2_2_4_64_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_25088:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_12544:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_73728:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_128:array<f32>;
@compute @workgroup_size(32,2,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 7 */
  var lidx0 = i32(lindex.x); /* 32 */
  var lidx1 = i32(lindex.y); /* 2 */
  var lidx2 = i32(lindex.z); /* 2 */
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<1u);
  var cast0 = bitcast<i32>(precast1);
  var alu0 = (lidx1*14);
  var alu1 = (gidx1*28);
  var alu2 = (lidx2+cast0);
  var alu3 = (alu2<13);
  var precast2 = gidx1;
  var precast3 = (bitcast<u32>(precast2)<<1u);
  var alu4 = ((lidx1+bitcast<i32>(precast3))<13);
  var alu5 = (0<(gidx1+lidx1));
  var alu6 = (0<(gidx0+lidx2));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var ridx1006 = 0; ridx1006 < 64; ridx1006++) {
    var alu11 = ((lidx0*2304)+(ridx1006*9));
    var val0 = data2_73728[alu11];
    var val1 = data2_73728[(alu11+1)];
    var val2 = data2_73728[(alu11+2)];
    var val3 = data2_73728[(alu11+3)];
    var val4 = data2_73728[(alu11+4)];
    var val5 = data2_73728[(alu11+5)];
    var val6 = data2_73728[(alu11+6)];
    var val7 = data2_73728[(alu11+7)];
    var val8 = data2_73728[(alu11+8)];
    var val9 = data2_73728[(alu11+576)];
    var val10 = data2_73728[(alu11+577)];
    var val11 = data2_73728[(alu11+578)];
    var val12 = data2_73728[(alu11+579)];
    var val13 = data2_73728[(alu11+580)];
    var val14 = data2_73728[(alu11+581)];
    var val15 = data2_73728[(alu11+582)];
    var val16 = data2_73728[(alu11+583)];
    var val17 = data2_73728[(alu11+584)];
    var val18 = data2_73728[(alu11+1152)];
    var val19 = data2_73728[(alu11+1153)];
    var val20 = data2_73728[(alu11+1154)];
    var val21 = data2_73728[(alu11+1155)];
    var val22 = data2_73728[(alu11+1156)];
    var val23 = data2_73728[(alu11+1157)];
    var val24 = data2_73728[(alu11+1158)];
    var val25 = data2_73728[(alu11+1159)];
    var val26 = data2_73728[(alu11+1160)];
    var val27 = data2_73728[(alu11+1728)];
    var val28 = data2_73728[(alu11+1729)];
    var val29 = data2_73728[(alu11+1730)];
    var val30 = data2_73728[(alu11+1731)];
    var val31 = data2_73728[(alu11+1732)];
    var val32 = data2_73728[(alu11+1733)];
    var val33 = data2_73728[(alu11+1734)];
    var val34 = data2_73728[(alu11+1735)];
    var val35 = data2_73728[(alu11+1736)];
    var alu12 = (alu2+alu1+alu0+(ridx1006*196));
    var val36 = data1_12544[alu12];
    var val37 = select(0.0f, data1_12544[(alu12+-15)], (alu5&alu6));
    var val38 = select(0.0f, data1_12544[(alu12+-14)], alu5);
    var val39 = select(0.0f, data1_12544[(alu12+-13)], (alu5&alu3));
    var val40 = select(0.0f, data1_12544[(alu12+-1)], alu6);
    var val41 = select(0.0f, data1_12544[(alu12+1)], alu3);
    var val42 = select(0.0f, data1_12544[(alu12+13)], (alu4&alu6));
    var val43 = select(0.0f, data1_12544[(alu12+14)], alu4);
    var val44 = select(0.0f, data1_12544[(alu12+15)], (alu4&alu3));
    acc0[0] = (acc0[0]+(val37*val0)+(val40*val3)+(val42*val6)+(val38*val1)+(val36*val4)+(val43*val7)+(val39*val2)+(val41*val5)+(val44*val8));
    acc0[1] = (acc0[1]+(val37*val9)+(val40*val12)+(val42*val15)+(val38*val10)+(val36*val13)+(val43*val16)+(val39*val11)+(val41*val14)+(val44*val17));
    acc0[2] = (acc0[2]+(val37*val18)+(val40*val21)+(val42*val24)+(val38*val19)+(val36*val22)+(val43*val25)+(val39*val20)+(val41*val23)+(val44*val26));
    acc0[3] = (acc0[3]+(val37*val27)+(val40*val30)+(val42*val33)+(val38*val28)+(val36*val31)+(val43*val34)+(val39*val29)+(val41*val32)+(val44*val35));
  }
  var precast4 = lidx0;
  var precast5 = (bitcast<u32>(precast4)<<2u);
  var cast1 = bitcast<i32>(precast5);
  var val45 = data3_128[cast1];
  var val46 = data3_128[(cast1+1)];
  var val47 = data3_128[(cast1+2)];
  var val48 = data3_128[(cast1+3)];
  var alu18 = (lidx2+cast0+alu1+(lidx0*784)+alu0);
  data0_25088[alu18] = (acc0[0]+val45);
  data0_25088[(alu18+196)] = (acc0[1]+val46);
  data0_25088[(alu18+392)] = (acc0[2]+val47);
  data0_25088[(alu18+588)] = (acc0[3]+val48);
}`;

const r_7_7_32_4_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_6272:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_25088:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 7 */
  var lidx0 = i32(lindex.x); /* 32 */
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<1u);
  var alu0 = (bitcast<i32>(precast1)+(gidx1*3584)+(lidx0*112));
  var val0 = data1_25088[alu0];
  var val1 = data1_25088[(alu0+1)];
  var val2 = data1_25088[(alu0+14)];
  var val3 = data1_25088[(alu0+15)];
  var val4 = data1_25088[(alu0+28)];
  var val5 = data1_25088[(alu0+29)];
  var val6 = data1_25088[(alu0+42)];
  var val7 = data1_25088[(alu0+43)];
  var val8 = data1_25088[(alu0+56)];
  var val9 = data1_25088[(alu0+57)];
  var val10 = data1_25088[(alu0+70)];
  var val11 = data1_25088[(alu0+71)];
  var val12 = data1_25088[(alu0+84)];
  var val13 = data1_25088[(alu0+85)];
  var val14 = data1_25088[(alu0+98)];
  var val15 = data1_25088[(alu0+99)];
  var alu1 = (gidx0+(gidx1*896)+(lidx0*28));
  var alu2 = select(0.0f,val0,(0.0f<val0));
  var alu3 = select(0.0f,val1,(0.0f<val1));
  var alu4 = select(0.0f,val2,(0.0f<val2));
  var alu5 = select(0.0f,val3,(0.0f<val3));
  data0_6272[alu1] = ((alu2+alu4+alu3+alu5)*0.25f);
  var alu7 = select(0.0f,val4,(0.0f<val4));
  var alu8 = select(0.0f,val5,(0.0f<val5));
  var alu9 = select(0.0f,val6,(0.0f<val6));
  var alu10 = select(0.0f,val7,(0.0f<val7));
  data0_6272[(alu1+7)] = ((alu7+alu9+alu8+alu10)*0.25f);
  var alu12 = select(0.0f,val8,(0.0f<val8));
  var alu13 = select(0.0f,val9,(0.0f<val9));
  var alu14 = select(0.0f,val10,(0.0f<val10));
  var alu15 = select(0.0f,val11,(0.0f<val11));
  data0_6272[(alu1+14)] = ((alu12+alu14+alu13+alu15)*0.25f);
  var alu17 = select(0.0f,val12,(0.0f<val12));
  var alu18 = select(0.0f,val13,(0.0f<val13));
  var alu19 = select(0.0f,val14,(0.0f<val14));
  var alu20 = select(0.0f,val15,(0.0f<val15));
  data0_6272[(alu1+21)] = ((alu17+alu19+alu18+alu20)*0.25f);
}`;

const r_8_4_4_8_784 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_128:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_6272:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_802816:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_128:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var acc1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var precast0 = lidx0;
  var precast1 = (bitcast<u32>(precast0)<<5u);
  var cast0 = bitcast<i32>(precast1);
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  var precast2 = gidx0;
  var precast3 = (bitcast<u32>(precast2)<<4u);
  var alu4 = (lidx0+bitcast<i32>(precast3));
  var val0 = data3_128[alu4];
  var alu5 = (alu4+4);
  var val1 = data3_128[alu5];
  var alu6 = (alu4+8);
  var val2 = data3_128[alu6];
  var alu7 = (alu4+12);
  var val3 = data3_128[alu7];
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var ridx3004 = 0; ridx3004 < 784; ridx3004++) {
    var precast4 = ridx3004;
    var precast5 = (bitcast<u32>(precast4)<<3u);
    var cast1 = bitcast<i32>(precast5);
    var val4 = data1_6272[(lidx1+cast1)];
    var alu12 = (lidx1+(gidx0*100352)+(lidx0*6272)+cast1);
    var val5 = data2_802816[alu12];
    var val6 = data2_802816[(alu12+25088)];
    var val7 = data2_802816[(alu12+50176)];
    var val8 = data2_802816[(alu12+75264)];
    acc0[0] = (acc0[0]+(val4*val5));
    acc0[1] = (acc0[1]+(val4*val6));
    acc0[2] = (acc0[2]+(val4*val7));
    acc0[3] = (acc0[3]+(val4*val8));
  }
  var precast6 = lidx1;
  var precast7 = (bitcast<u32>(precast6)<<2u);
  var alu18 = (cast0+bitcast<i32>(precast7));
  temp0[alu18] = acc0[0];
  temp0[(alu18+1)] = acc0[1];
  temp0[(alu18+2)] = acc0[2];
  temp0[(alu18+3)] = acc0[3];
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    for (var ridx1003 = 0; ridx1003 < 8; ridx1003++) {
      var precast8 = ridx1003;
      var precast9 = (bitcast<u32>(precast8)<<2u);
      var alu25 = (cast0+bitcast<i32>(precast9));
      var val9 = temp0[alu25];
      var val10 = temp0[(alu25+1)];
      var val11 = temp0[(alu25+2)];
      var val12 = temp0[(alu25+3)];
      acc1[0] = (acc1[0]+val9);
      acc1[1] = (acc1[1]+val10);
      acc1[2] = (acc1[2]+val11);
      acc1[3] = (acc1[3]+val12);
    }
    var alu31 = (acc1[0]+val0);
    var alu32 = (acc1[1]+val1);
    var alu33 = (acc1[2]+val2);
    var alu34 = (acc1[3]+val3);
    var alu35 = select(0.0f,alu31,(0.0f<alu31));
    data0_128[alu4] = alu35;
    var alu37 = select(0.0f,alu32,(0.0f<alu32));
    data0_128[alu5] = alu37;
    var alu39 = select(0.0f,alu33,(0.0f<alu33));
    data0_128[alu6] = alu39;
    var alu41 = select(0.0f,alu34,(0.0f<alu34));
    data0_128[alu7] = alu41;
  }
}`;

const r_10_16_8n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_10:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_128:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1280:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_10:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 10 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc1[0] = 0.0f;
  var val0 = data3_10[gidx0];
  var precast0 = lidx0;
  var precast1 = (bitcast<u32>(precast0)<<3u);
  var cast0 = bitcast<i32>(precast1);
  acc0[0] = 0.0f;
  var precast2 = gidx0;
  var precast3 = (bitcast<u32>(precast2)<<7u);
  for (var ridx3002 = 0; ridx3002 < 8; ridx3002++) {
    var val1 = data1_128[(cast0+ridx3002)];
    var val2 = data2_1280[(bitcast<i32>(precast3)+cast0+ridx3002)];
    acc0[0] = (acc0[0]+(val1*val2));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    for (var ridx1001 = 0; ridx1001 < 16; ridx1001++) {
      var val3 = temp0[ridx1001];
      acc1[0] = (acc1[0]+val3);
    }
    data0_10[gidx0] = (acc1[0]+val0);
  }
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 100352);;
    const input0 = createEmptyBuf(device, 3136);;
    const buf_1 = createWeightBuf(device, 1152, getTensorBuffer(safetensor, metadata['layers.1.weight']));
    const buf_2 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.1.bias']));
    const buf_3 = createEmptyBuf(device, 200704);;
    const buf_4 = createWeightBuf(device, 73728, getTensorBuffer(safetensor, metadata['layers.3.weight']));
    const buf_5 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.3.bias']));
    const buf_6 = createEmptyBuf(device, 50176);;
    const buf_7 = createWeightBuf(device, 294912, getTensorBuffer(safetensor, metadata['layers.6.weight']));
    const buf_8 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['layers.6.bias']));
    const buf_9 = createEmptyBuf(device, 25088);;
    const buf_10 = createEmptyBuf(device, 512);;
    const buf_11 = createWeightBuf(device, 3211264, getTensorBuffer(safetensor, metadata['layers.10.weight']));
    const buf_12 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['layers.10.bias']));
    const output0 = createEmptyBuf(device, 40);;
    const buf_13 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['layers.12.weight']));
    const buf_14 = createWeightBuf(device, 40, getTensorBuffer(safetensor, metadata['layers.12.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_7_7_8_4_4_4_3_3, r_7_7_16_4_4_4_32_3_3, r_7_7_32_2_4_2_2, r_7_7_32_2_2_4_64_3_3, r_7_7_32_4_2_2, r_8_4_4_8_784, r_10_16_8n1];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1, buf_2], [7, 7, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_3, buf_0, buf_4, buf_5], [7, 7, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_6, buf_3], [7, 7, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_0, buf_6, buf_7, buf_8], [7, 7, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_9, buf_0], [7, 7, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_10, buf_9, buf_11, buf_12], [8, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [output0, buf_10, buf_13, buf_14], [10, 1, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default mnist_convnet;
