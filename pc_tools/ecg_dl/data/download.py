#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT-BIH Arrhythmia Database 下载器
使用 wfdb 库从 PhysioNet 官方源下载
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIT_BIH_LOCAL_DIR, RAW_DATA_DIR, MIT_BIH_RECORDS


def download_mit_bih(force_redownload: bool = False) -> Path:
    """
    使用 wfdb 从 PhysioNet 下载 MIT-BIH 数据集
    
    Args:
        force_redownload: 如果 True, 重新下载
        
    Returns:
        数据集本地路径
    """
    import wfdb
    
    # wfdb.dl_database 会自动创建目录
    print(f"[下载] 从 PhysioNet 下载 MIT-BIH 数据集...")
    print(f"[下载] 目标: {MIT_BIH_LOCAL_DIR}")
    print(f"[下载] 包含 {len(MIT_BIH_RECORDS)} 条记录")
    
    if force_redownload and MIT_BIH_LOCAL_DIR.exists():
        import shutil
        shutil.rmtree(MIT_BIH_LOCAL_DIR)
        print(f"[下载] 已删除旧目录")
    
    if MIT_BIH_LOCAL_DIR.exists():
        dat_files = list(MIT_BIH_LOCAL_DIR.glob("*.dat"))
        if len(dat_files) >= 10:
            print(f"[下载] 数据集已存在 ({len(dat_files)} 个 .dat 文件)")
            return MIT_BIH_LOCAL_DIR
    
    try:
        wfdb.dl_database(
            db_dir='mitdb',
            dl_dir=str(MIT_BIH_LOCAL_DIR),
            records='all',
            annotators='all',
            keep_subdirs=False,
            overwrite=force_redownload
        )
        print(f"[下载] 下载完成: {MIT_BIH_LOCAL_DIR}")
        
    except Exception as e:
        print(f"[下载] wfdb 下载失败: {e}")
        print("[下载] 尝试替代方案...")
        
        # 备用方案: PhysioNet 直接下载
        import urllib.request
        base_url = "https://physionet.org/files/mitdb/1.0.0/"
        
        MIT_BIH_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        
        for record in MIT_BIH_RECORDS:
            for ext in ['hea', 'dat', 'atr']:
                filename = f"{record}.{ext}"
                url = base_url + filename
                filepath = MIT_BIH_LOCAL_DIR / filename
                
                if filepath.exists():
                    continue
                    
                print(f"[下载]   {filename}...", end=" ")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print("OK")
                except Exception as e2:
                    print(f"FAILED: {e2}")
        
        print(f"[下载] 备用下载完成")
    
    return MIT_BIH_LOCAL_DIR


def verify_dataset() -> bool:
    """
    验证数据集完整性
    """
    if not MIT_BIH_LOCAL_DIR.exists():
        print("[验证] 数据集目录不存在")
        return False
    
    dat_files = sorted(MIT_BIH_LOCAL_DIR.glob("*.dat"))
    hea_files = sorted(MIT_BIH_LOCAL_DIR.glob("*.hea"))
    atr_files = sorted(MIT_BIH_LOCAL_DIR.glob("*.atr"))
    
    print(f"[验证] .dat 文件: {len(dat_files)} 个")
    print(f"[验证] .hea 文件: {len(hea_files)} 个")
    print(f"[验证] .atr 文件: {len(atr_files)} 个")
    
    if len(dat_files) >= 10:
        print(f"[验证] 数据集基本完整")
        # 打印几条记录
        print(f"[验证] 前5条记录:")
        for f in dat_files[:5]:
            size_kb = f.stat().st_size / 1024
            print(f"       {f.name} ({size_kb:.0f} KB)")
        return True
    else:
        print(f"[验证] 数据不完整")
        return False


def download_minimal_test_set() -> Path:
    """
    下载最小测试集 (仅 100, 105, 200 三条记录)
    用于快速测试整个 pipeline
    """
    import urllib.request
    base_url = "https://physionet.org/files/mitdb/1.0.0/"
    
    test_records = ['100', '105', '200']
    test_dir = RAW_DATA_DIR / "mitdb_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[下载] 下载最小测试集到 {test_dir}")
    
    for record in test_records:
        for ext in ['hea', 'dat', 'atr']:
            filename = f"{record}.{ext}"
            url = base_url + filename
            filepath = test_dir / filename
            
            if filepath.exists():
                print(f"[下载]   {filename} 已存在")
                continue
                
            print(f"[下载]   {filename}...", end=" ")
            try:
                urllib.request.urlretrieve(url, filepath)
                size_kb = filepath.stat().st_size / 1024
                print(f"OK ({size_kb:.0f} KB)")
            except Exception as e:
                print(f"FAILED: {e}")
    
    print(f"[下载] 最小测试集下载完成: {test_dir}")
    return test_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="下载 MIT-BIH 心律失常数据集")
    parser.add_argument("--force", action="store_true", help="重新下载")
    parser.add_argument("--test-only", action="store_true", help="仅下载最小测试集 (3条记录)")
    args = parser.parse_args()
    
    if args.test_only:
        download_minimal_test_set()
    else:
        download_mit_bih(force_redownload=args.force)
        verify_dataset()