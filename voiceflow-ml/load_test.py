#!/usr/bin/env python3
"""
Load test script for VoiceFlow Inference Server

Tests the inference server with concurrent requests and measures:
- Request latency (P50, P95, P99)
- Throughput (requests/second)
- Error rate

Usage:
    python load_test.py --url http://localhost:3000 --requests 1000 --concurrency 10
"""

import argparse
import time
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict
import json


async def send_inference_request(
    session: aiohttp.ClientSession,
    url: str,
    audio_data: List[float]
) -> Dict:
    """Send single inference request"""
    start_time = time.perf_counter()
    
    try:
        async with session.post(
            f"{url}/infer",
            json={"audio": audio_data, "sample_rate": 48000},
            timeout=aiohttp.ClientTimeout(total=5.0)
        ) as response:
            if response.status == 200:
                data = await response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "success": True,
                    "latency_ms": latency_ms,
                    "server_latency_ms": data.get("latency_ms", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status}",
                    "latency_ms": (time.perf_counter() - start_time) * 1000
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency_ms": (time.perf_counter() - start_time) * 1000
        }


async def load_test(
    url: str,
    num_requests: int,
    concurrency: int
) -> Dict:
    """Run load test with specified parameters"""
    
    # Generate test audio (1 second at 48kHz)
    audio_data = np.random.randn(48000).astype(np.float32).tolist()
    
    print(f"""
==================================================================
              VoiceFlow Load Test                               
==================================================================

Target: {url}
Total Requests: {num_requests}
Concurrency: {concurrency}
Audio Size: 48000 samples (1 second @ 48kHz)

Starting test...
""")
    
    # Create session
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        
        # Check server health
        try:
            async with session.get(f"{url}/health") as response:
                if response.status != 200:
                    print(f"❌ Server not healthy: HTTP {response.status}")
                    return {}
                print("✅ Server health check passed\n")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return {}
        
        # Run load test
        start_time = time.perf_counter()
        results = []
        
        # Create batches
        batch_size = concurrency
        num_batches = (num_requests + batch_size - 1) // batch_size
        
        for batch_num in range(num_batches):
            # Create tasks for this batch
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, num_requests)
            batch_tasks = []
            
            for i in range(batch_start, batch_end):
                task = send_inference_request(session, url, audio_data)
                batch_tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Progress update
            completed = len(results)
            progress = (completed / num_requests) * 100
            print(f"Progress: {completed}/{num_requests} ({progress:.1f}%)", end="\r")
        
        total_time = time.perf_counter() - start_time
        
    print(f"\n\n✅ Load test complete! ({total_time:.2f}s)")
    
    # Analyze results
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    
    if successes:
        latencies = [r["latency_ms"] for r in successes]
        server_latencies = [r["server_latency_ms"] for r in successes if "server_latency_ms" in r]
        
        latencies.sort()
        if server_latencies:
            server_latencies.sort()
        
        stats = {
            "total_requests": num_requests,
            "successful": len(successes),
            "failed": len(failures),
            "success_rate": len(successes) / num_requests * 100,
            "total_time_s": total_time,
            "throughput_rps": len(successes) / total_time,
            "latency": {
                "min": latencies[0],
                "median": latencies[len(latencies) // 2],
                "p95": latencies[int(len(latencies) * 0.95)],
                "p99": latencies[int(len(latencies) * 0.99)],
                "max": latencies[-1]
            }
        }
        
        if server_latencies:
            stats["server_latency"] = {
                "min": server_latencies[0],
                "median": server_latencies[len(server_latencies) // 2],
                "p95": server_latencies[int(len(server_latencies) * 0.95)],
                "p99": server_latencies[int(len(server_latencies) * 0.99)],
                "max": server_latencies[-1]
            }
    else:
        stats = {
            "total_requests": num_requests,
            "successful": 0,
            "failed": len(failures),
            "errors": [r.get("error") for r in failures[:10]]
        }
    
    return stats


def print_results(stats: Dict):
    """Print formatted test results"""
    
    if stats.get("successful", 0) == 0:
        print("\n❌ All requests failed!")
        print(f"Errors: {stats.get('errors', [])}")
        return
    
    print(f"""
==================================================================
                    Test Results                                
==================================================================

Requests:
  * Total:       {stats['total_requests']}
  * Successful:  {stats['successful']} ({stats['success_rate']:.1f}%)
  * Failed:      {stats['failed']}

Performance:
  * Total Time:  {stats['total_time_s']:.2f}s
  * Throughput:  {stats['throughput_rps']:.1f} req/s

End-to-End Latency (Client → Server → Client):
  * Min:         {stats['latency']['min']:.2f} ms
  * Median:      {stats['latency']['median']:.2f} ms
  * P95:         {stats['latency']['p95']:.2f} ms
  * P99:         {stats['latency']['p99']:.2f} ms
  * Max:         {stats['latency']['max']:.2f} ms
""")
    
    if "server_latency" in stats:
        print(f"""Model Inference Latency (Server-side):
  * Min:         {stats['server_latency']['min']:.2f} ms
  * Median:      {stats['server_latency']['median']:.2f} ms
  * P95:         {stats['server_latency']['p95']:.2f} ms
  * P99:         {stats['server_latency']['p99']:.2f} ms
  * Max:         {stats['server_latency']['max']:.2f} ms
""")
    
    # SLA validation
    end_to_end_p99 = stats['latency']['p99']
    model_p99 = stats.get('server_latency', {}).get('p99', 0)
    
    print("SLA Validation:")
    if model_p99 > 0:
        model_status = "✅ PASS" if model_p99 < 10 else "❌ FAIL"
        print(f"  * Model P99 < 10ms:  {model_status} ({model_p99:.2f}ms)")
    
    e2e_status = "PASS" if end_to_end_p99 < 100 else "FAIL"
    print(f"  * End-to-End P99 < 100ms: {e2e_status} ({end_to_end_p99:.2f}ms)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Load test for VoiceFlow Inference Server')
    parser.add_argument('--url', default='http://localhost:3000', help='Server URL')
    parser.add_argument('--requests', type=int, default=100, help='Total number of requests')
    parser.add_argument('--concurrency', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Run load test
    stats = asyncio.run(load_test(args.url, args.requests, args.concurrency))
    
    # Print results
    print_results(stats)
    
    # Save to file if requested
    if args.output and stats:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
