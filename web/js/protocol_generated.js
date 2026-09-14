/* GENERATED FILE - DO NOT EDIT.
 * Source: protocol/ecg_proto.json
 * Regenerate: python scripts/gen_protocol_constants.py */
'use strict';
(function (global) {
  var P = {
    PROTO_VERSION: 2,
    FW_VERSION: '2.0.0',
    CSV_COLUMNS_V1: 9,
    CSV_COLUMNS_V2: 10,
    ECGR_HEADER_SIZE: 32,
    ECGR_VERSION_CURRENT: 2,
    ECGR_SCALE_TO_VOLTS: 8000.0,
    ECGR_FLAG_HAS_ABNORMAL_BITMAP: 0x01,
    ECGR_RESERVED0_ABNORMAL_IS_ASRC: 0x01,
    ASRC: {
      LEADOFF: 0x10,
      VF: 0x08,
      FLAT: 0x04,
      RS: 0x02,
      AI: 0x01,
    },
    ASRC_META: [
    { bit: 0x10, name: 'LEADOFF', label: '电极脱落', level: 'warning' },
    { bit: 0x08, name: 'VF', label: 'VF / VT 疑似', level: 'critical' },
    { bit: 0x04, name: 'FLAT', label: '时间停搏（疑似电极脱落）', level: 'critical' },
    { bit: 0x02, name: 'RS', label: '停搏 / 过缓过速', level: 'critical' },
    { bit: 0x01, name: 'AI', label: 'AI 异常', level: 'info' }
    ],
    primaryAsrc: function (asrc) {
      for (var i = 0; i < P.ASRC_META.length; i++) {
        if (asrc & P.ASRC_META[i].bit) return P.ASRC_META[i];
      }
      return null;
    }
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = P;
  global.ECGProtocol = P;
})(typeof window !== 'undefined' ? window : globalThis);
