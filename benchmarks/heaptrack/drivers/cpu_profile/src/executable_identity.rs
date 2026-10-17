//! Executable identity, excluding only documented linker bookkeeping.
//! PE layout: https://learn.microsoft.com/windows/win32/debug/pe-format

/// Preserve code, data, layout and unknown fields. Non-PE files remain byte exact.
/// A truncated MZ/PE header is an error, never a partially normalized identity.
pub(crate) fn normalized(bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut result = bytes.to_vec();
    if !bytes.starts_with(b"MZ") {
        return Ok(result);
    }
    let read32 = |offset: usize| -> Result<usize, &'static str> {
        let value = bytes
            .get(offset..offset.checked_add(4).ok_or("PE offset overflow")?)
            .ok_or("Truncated PE header")?;
        Ok(u32::from_le_bytes(value.try_into().unwrap()) as usize)
    };
    let pe = read32(0x3c)?;
    if bytes
        .get(pe..pe.checked_add(24).ok_or("PE offset overflow")?)
        .is_none_or(|header| !header.starts_with(b"PE\0\0"))
    {
        return Err("Invalid PE signature/header");
    }
    // COFF TimeDateStamp changes when MSVC relinks identical code.
    result[pe + 8..pe + 12].fill(0);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::normalized;
    fn pe() -> Vec<u8> {
        let mut bytes = vec![0; 512];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[0x3c..0x40].copy_from_slice(&128u32.to_le_bytes());
        bytes[128..132].copy_from_slice(b"PE\0\0");
        bytes[400] = 0xc3;
        bytes
    }
    #[test]
    fn relink_timestamp_does_not_change_key_material() {
        let first = pe();
        let mut second = first.clone();
        second[136..140].copy_from_slice(&123456u32.to_le_bytes());
        assert_ne!(first, second);
        assert_eq!(normalized(&first).unwrap(), normalized(&second).unwrap());
        second[400] ^= 1;
        assert_ne!(normalized(&first).unwrap(), normalized(&second).unwrap());
        second = first.clone();
        second[132] ^= 1; // Machine/architecture is significant too.
        assert_ne!(normalized(&first).unwrap(), normalized(&second).unwrap());
    }
    #[test]
    fn non_pe_and_malformed_images_are_not_silently_rewritten() {
        assert_eq!(normalized(b"\x7fELFpayload").unwrap(), b"\x7fELFpayload");
        assert!(normalized(b"MZ").is_err());
        let mut bytes = pe();
        bytes[0x3c..0x40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(normalized(&bytes).is_err());
        let mut bytes = pe();
        bytes[128] = 0;
        assert!(normalized(&bytes).is_err());
    }
}
