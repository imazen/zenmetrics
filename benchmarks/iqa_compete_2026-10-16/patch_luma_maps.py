#!/usr/bin/env python3
"""Expose existing grayscale map kernels only inside an owned benchmark snapshot."""
import argparse,hashlib,json,pathlib
P=pathlib.Path
WRAPPERS={
'fsim':r'''
/// Benchmark-snapshot adapter to the existing luma ingress and native map kernel.
#[doc(hidden)]
pub fn iqa_compete_luma_map(reference: &[u8], distorted: &[u8], width: usize, height: usize, stride: usize) -> Result<(f64, alloc::vec::Vec<f32>), Error> {
    check_len_rgb(reference, width, height, stride)?;
    check_len_rgb(distorted, width, height, stride)?;
    let factor = kernel::downsample_factor(width, height);
    let mut map = alloc::vec![0.0; width.div_ceil(factor) * height.div_ceil(factor)];
    let (score, w, h) = kernel::fsim_luma8_planes(reference, distorted, width, height, stride, Some(&mut map));
    map.truncate(w * h);
    Ok((score, map))
}
''',
'haarpsi':r'''
/// Benchmark-snapshot adapter to the existing luma ingress and native map kernel.
#[doc(hidden)]
pub fn iqa_compete_luma_map(reference: &[u8], distorted: &[u8], width: usize, height: usize, stride: usize) -> Result<(f64, alloc::vec::Vec<f32>), Error> {
    check_rgb(reference, distorted, width, height, stride)?;
    let pr = kernel::plane_from_rgb8(reference, width, height, stride, Y_COEF, true);
    let pd = kernel::plane_from_rgb8(distorted, width, height, stride, Y_COEF, true);
    let mut map = alloc::vec![0.0; pr.w * pr.h];
    let score = kernel::haarpsi_core(&pr, &pd, None, Some(&mut map));
    Ok((score, map))
}
'''}

def patch(snapshot,original,receipt):
 rows=[]
 for crate,wrapper in WRAPPERS.items():
  relative=P('crates')/crate/'src/lib.rs';path=snapshot/relative;source=original/relative
  if path.samefile(source):raise ValueError('Snapshot aliases the original checkout')
  before=path.read_bytes()
  if before!=source.read_bytes():raise ValueError('Unexpected prior kernel adaptation: '+str(relative))
  path.write_bytes(before+wrapper.encode())
  rows.append(dict(path=str(relative),original_sha256=hashlib.sha256(before).hexdigest(),adapted_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 data=dict(adapter_sha256=hashlib.sha256(P(__file__).read_bytes()).hexdigest(),files=rows,
  scope='Owned snapshot only; existing luma ingress and native map kernels reused. No shipped API or scoring arithmetic changed.')
 content=json.dumps(data,indent=2)+'\n'
 if receipt.exists() and receipt.read_text()!=content:
  backup=receipt.with_name(receipt.stem+'-'+hashlib.sha256(receipt.read_bytes()).hexdigest()[:16]+'.bak.json')
  if not backup.exists():receipt.rename(backup)
 receipt.write_text(content)
 return data

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['snapshot','original','receipt']:p.add_argument(name,type=P)
 a=p.parse_args();print(json.dumps(patch(a.snapshot,a.original,a.receipt)))
