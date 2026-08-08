import 'package:flutter_test/flutter_test.dart';

import 'package:ecg_app/services/csv_parser.dart';

/**
 * @brief BLE/串口 CSV 行解析单元测试
 *
 * 字段顺序: clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal_flag,confidence
 * 测试覆盖：完整行 / 部分行默认值 / 异常行容错 / 索引正确性
 */
void main() {
  group('parseEcgCsvLine', () {
    test('完整 9 列行解析所有字段', () {
      final sample = parseEcgCsvLine('0.253,-0.187,0.241,75,75,0.87,0,0,0.012');
      expect(sample, isNotNull);
      expect(sample!.clean, closeTo(0.253, 1e-9));
      expect(sample.noisy, closeTo(-0.187, 1e-9));
      expect(sample.filtered, closeTo(0.241, 1e-9));
      expect(sample.bpm, 75);
      expect(sample.abnormal, 0);
      expect(sample.confidence, closeTo(0.012, 1e-9));
    });

    test('abnormal 位于索引 7（第 8 列）、confidence 位于索引 8（第 9 列）', () {
      final sample = parseEcgCsvLine('0.1,0.2,0.3,60,60,0.9,1,1,0.87');
      expect(sample, isNotNull);
      expect(sample!.bpm, 60);
      // 第 7 列 motion=1 不得影响 abnormal
      expect(sample.abnormal, 1);
      expect(sample.confidence, closeTo(0.87, 1e-9));
    });

    test('3 列行只解析前三通道，其余取默认值', () {
      final sample = parseEcgCsvLine('0.5,-0.2,0.4');
      expect(sample, isNotNull);
      expect(sample!.clean, closeTo(0.5, 1e-9));
      expect(sample.noisy, closeTo(-0.2, 1e-9));
      expect(sample.filtered, closeTo(0.4, 1e-9));
      expect(sample.bpm, 0);
      expect(sample.abnormal, 0);
      expect(sample.confidence, 0.0);
    });

    test('部分可选列存在时按索引解析（如 4 列行）', () {
      final sample = parseEcgCsvLine('0.1,0.2,0.3,72');
      expect(sample, isNotNull);
      expect(sample!.bpm, 72);
      expect(sample.abnormal, 0);
      expect(sample.confidence, 0.0);
    });

    test('空行与不足 3 列返回 null', () {
      expect(parseEcgCsvLine(''), isNull);
      expect(parseEcgCsvLine('   '), isNull);
      expect(parseEcgCsvLine('0.1,0.2'), isNull);
    });

    test('前三列非数字返回 null', () {
      expect(parseEcgCsvLine('abc,def,ghi'), isNull);
      expect(parseEcgCsvLine('0.1,x,0.3'), isNull);
    });

    test('可选列非数字时容错为默认值（tryParse 语义）', () {
      final sample = parseEcgCsvLine('0.1,0.2,0.3,abc,0,0,0,1,xyz');
      expect(sample, isNotNull);
      expect(sample!.bpm, 0);
      expect(sample.abnormal, 1); // 第 8 列可解析
      expect(sample.confidence, 0.0); // 第 9 列无法解析 → 默认 0.0
    });

    test('容忍首尾空白与字段内空格', () {
      final sample = parseEcgCsvLine(' 0.1, 0.2 ,0.3 ,70');
      expect(sample, isNotNull);
      expect(sample!.clean, closeTo(0.1, 1e-9));
      expect(sample.bpm, 70);
    });
  });
}
