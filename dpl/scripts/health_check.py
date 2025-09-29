#!/usr/bin/env python3
"""
DPL Tailscale健康检查脚本
用于监控网络连通性和服务可用性
"""

import asyncio
import httpx
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# 配置节点信息 - 已更新为实际IP地址
NODES = [
    {"id": 1, "ip": "100.76.208.127", "monitor_port": 8001, "llm_port": 1234},
    #{"id": 2, "ip": "100.118.49.57", "monitor_port": 8001, "llm_port": 1234},
]

async def check_tailscale_status():
    """检查Tailscale服务状态"""
    try:
        result = subprocess.run(['tailscale', 'status'], 
                              capture_output=True, text=True, check=True)
        return True, "Tailscale运行正常"
    except subprocess.CalledProcessError as e:
        return False, f"Tailscale状态异常: {e}"
    except FileNotFoundError:
        return False, "Tailscale客户端未安装"

async def check_network_connectivity(ip):
    """检查网络连通性"""
    try:
        result = subprocess.run(['ping', '-c', '1', '-W', '3', ip], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

async def check_monitor_api(ip, port):
    """检查监控API可用性"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://{ip}:{port}/status")
            return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

async def check_llm_api(ip, port):
    """检查LLM API可用性"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://{ip}:{port}/v1/models")
            return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

async def health_check():
    """执行完整的健康检查"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔍 [{timestamp}] 开始健康检查...")
    
    # 检查Tailscale状态
    ts_status, ts_msg = await check_tailscale_status()
    print(f"{'✅' if ts_status else '❌'} Tailscale状态: {ts_msg}")
    
    if not ts_status:
        return
    
    # 检查各节点
    all_healthy = True
    
    for node in NODES:
        node_id = node["id"]
        ip = node["ip"]
        monitor_port = node["monitor_port"]
        llm_port = node["llm_port"]
        
        print(f"\n📡 检查节点 {node_id} ({ip}):")
        
        # 网络连通性
        network_ok = await check_network_connectivity(ip)
        print(f"  {'✅' if network_ok else '❌'} 网络连通性: {'正常' if network_ok else '异常'}")
        
        if not network_ok:
            all_healthy = False
            continue
        
        # 监控API
        monitor_ok, monitor_data = await check_monitor_api(ip, monitor_port)
        print(f"  {'✅' if monitor_ok else '❌'} 监控API: {'正常' if monitor_ok else '异常'}")
        
        if monitor_ok and isinstance(monitor_data, dict):
            locked = monitor_data.get("locked", False)
            model_id = monitor_data.get("model_id", "Unknown")
            cpu_usage = monitor_data.get("cpu_usage_percent", 0)
            gpu_info = monitor_data.get("gpu", {})
            
            print(f"    🔒 锁定状态: {'是' if locked else '否'}")
            print(f"    🤖 模型: {model_id}")
            print(f"    💻 CPU使用率: {cpu_usage}%")
            
            if gpu_info.get("available"):
                gpu_util = gpu_info.get("utilization_percent", 0)
                gpu_temp = gpu_info.get("temperature_celsius", 0)
                print(f"    🎮 GPU使用率: {gpu_util}%")
                print(f"    🌡️  GPU温度: {gpu_temp}°C")
        else:
            all_healthy = False
        
        # LLM API  
        llm_ok, llm_data = await check_llm_api(ip, llm_port)
        print(f"  {'✅' if llm_ok else '❌'} LLM API: {'正常' if llm_ok else '异常'}")
        
        if not (monitor_ok and llm_ok):
            all_healthy = False
    
    print(f"\n🏁 总体状态: {'✅ 全部正常' if all_healthy else '❌ 存在异常'}")
    return all_healthy

async def continuous_monitoring(interval=60):
    """持续监控模式"""
    print(f"🔄 启动持续监控模式 (间隔: {interval}秒)")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            await health_check()
            print(f"\n⏱️  等待 {interval} 秒后进行下次检查...\n" + "="*60)
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DPL Tailscale健康检查工具")
    parser.add_argument("--continuous", "-c", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", "-i", type=int, default=60, help="监控间隔(秒)")
    
    args = parser.parse_args()
    
    if args.continuous:
        asyncio.run(continuous_monitoring(args.interval))
    else:
        asyncio.run(health_check())

if __name__ == "__main__":
    main() 