//! Executable identity, excluding measured MSVC relink bookkeeping.
//! PE layout: https://learn.microsoft.com/windows/win32/debug/pe-format

fn u16_at(bytes: &[u8], offset: usize) -> Result<usize, &'static str> {
    let end = offset.checked_add(2).ok_or("PE offset overflow")?;
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..end)
            .ok_or("Truncated PE field")?
            .try_into()
            .unwrap(),
    ) as usize)
}
fn u32_at(bytes: &[u8], offset: usize) -> Result<usize, &'static str> {
    let end = offset.checked_add(4).ok_or("PE offset overflow")?;
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..end)
            .ok_or("Truncated PE field")?
            .try_into()
            .unwrap(),
    ) as usize)
}

/// Non-PE files remain byte exact. For PE images, ignore COFF/debug timestamps
/// and RSDS symbol-file GUIDs, retaining code, data, section layout, PDB paths,
/// ages and all other bytes. Invalid/truncated directories fail closed.
pub(crate) fn normalized(bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut result = bytes.to_vec();
    if !bytes.starts_with(b"MZ") {
        return Ok(result);
    }
    let pe = u32_at(bytes, 0x3c)?;
    let opt = pe.checked_add(24).ok_or("PE offset overflow")?;
    if bytes.get(pe..opt).is_none_or(|h| !h.starts_with(b"PE\0\0")) {
        return Err("Invalid PE signature/header");
    }
    let optional_size = u16_at(bytes, pe + 20)?;
    let sections = opt.checked_add(optional_size).ok_or("PE offset overflow")?;
    bytes
        .get(opt..sections)
        .ok_or("Truncated PE optional header")?;
    let directories = match u16_at(bytes, opt)? {
        0x10b => 96,  // PE32
        0x20b => 112, // PE32+
        _ => return Err("Unsupported PE optional header"),
    };
    if optional_size < directories {
        return Err("Truncated PE optional header");
    }
    result[pe + 8..pe + 12].fill(0); // COFF TimeDateStamp
    if u32_at(bytes, opt + directories - 4)? <= 6 {
        return Ok(result);
    }
    if optional_size < directories + 7 * 8 {
        return Err("Truncated PE debug directory pointer");
    }
    let rva = u32_at(bytes, opt + directories + 6 * 8)?;
    let size = u32_at(bytes, opt + directories + 6 * 8 + 4)?;
    if rva == 0 && size == 0 {
        return Ok(result);
    }
    if size == 0 || size % 28 != 0 {
        return Err("Invalid PE debug directory size");
    }
    let mut debug = None;
    for index in 0..u16_at(bytes, pe + 6)? {
        let section = sections
            .checked_add(index * 40)
            .ok_or("PE offset overflow")?;
        let va = u32_at(bytes, section + 12)?;
        let raw_size = u32_at(bytes, section + 16)?;
        let raw = u32_at(bytes, section + 20)?;
        if let Some(delta) = rva.checked_sub(va)
            && delta.checked_add(size).is_some_and(|end| end <= raw_size)
        {
            debug = Some(raw.checked_add(delta).ok_or("PE offset overflow")?);
            break;
        }
    }
    let debug = debug.ok_or("PE debug directory outside sections")?;
    let end = debug.checked_add(size).ok_or("PE offset overflow")?;
    bytes
        .get(debug..end)
        .ok_or("Truncated PE debug directory")?;
    for entry in (debug..end).step_by(28) {
        result[entry + 4..entry + 8].fill(0); // IMAGE_DEBUG_DIRECTORY.TimeDateStamp
        if u32_at(bytes, entry + 12)? == 2 {
            // IMAGE_DEBUG_TYPE_CODEVIEW
            let length = u32_at(bytes, entry + 16)?;
            let offset = u32_at(bytes, entry + 24)?;
            let end = offset.checked_add(length).ok_or("PE offset overflow")?;
            let payload = bytes.get(offset..end).ok_or("Truncated CodeView data")?;
            if payload.starts_with(b"RSDS") {
                if length < 24 {
                    return Err("Truncated RSDS data");
                }
                result[offset + 4..offset + 20].fill(0); // PDB GUID, not code
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::normalized;
    fn put32(bytes: &mut [u8], at: usize, value: u32) {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn pe(plus: bool) -> Vec<u8> {
        let mut b = vec![0; 1024];
        b[..2].copy_from_slice(b"MZ");
        put32(&mut b, 0x3c, 128);
        b[128..132].copy_from_slice(b"PE\0\0");
        b[134..136].copy_from_slice(&1u16.to_le_bytes());
        b[148..150].copy_from_slice(&240u16.to_le_bytes());
        b[152..154].copy_from_slice(&(if plus { 0x20bu16 } else { 0x10bu16 }).to_le_bytes());
        let directories = 152 + if plus { 112 } else { 96 };
        put32(&mut b, directories - 4, 16);
        put32(&mut b, directories + 48, 0x1000);
        put32(&mut b, directories + 52, 84);
        put32(&mut b, 392 + 12, 0x1000); // section RVA
        put32(&mut b, 392 + 16, 512); // raw size
        put32(&mut b, 392 + 20, 512); // raw offset
        put32(&mut b, 512 + 12, 2);
        put32(&mut b, 512 + 16, 40);
        put32(&mut b, 512 + 24, 640);
        b[640..644].copy_from_slice(b"RSDS");
        put32(&mut b, 660, 1); // PDB age
        b[664..680].copy_from_slice(b"iqa_compete.pdb\0");
        b[900] = 0xc3;
        b
    }
    #[test]
    fn relink_metadata_does_not_change_key_material_but_code_and_data_do() {
        for plus in [false, true] {
            let first = pe(plus);
            let mut second = first.clone();
            for offset in [136, 516, 544, 572] {
                put32(&mut second, offset, 123456);
            }
            second[644..660].fill(42);
            assert_ne!(first, second);
            assert_eq!(normalized(&first).unwrap(), normalized(&second).unwrap());
            for offset in [132, 900, 660, 664, 600] {
                // machine, code, age, path, other data
                let mut changed = second.clone();
                changed[offset] ^= 1;
                assert_ne!(normalized(&first).unwrap(), normalized(&changed).unwrap());
            }
        }
    }
    #[test]
    fn non_pe_and_malformed_images_are_not_silently_rewritten() {
        assert_eq!(normalized(b"\x7fELFpayload").unwrap(), b"\x7fELFpayload");
        assert!(normalized(b"MZ").is_err());
        for offset in [0x3c, 392 + 20, 512 + 24] {
            let mut b = pe(true);
            put32(&mut b, offset, u32::MAX);
            assert!(normalized(&b).is_err());
        }
        let mut b = pe(true);
        b[128] = 0;
        assert!(normalized(&b).is_err());
        let mut b = pe(true);
        put32(&mut b, 512 + 16, 8);
        assert!(normalized(&b).is_err());
    }
}
