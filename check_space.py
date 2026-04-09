import urllib.request, json

base = "https://dwraj-email-triage-openenv.hf.space"

# Reset
reset_data = json.dumps({"task_id": "easy"}).encode()
req = urllib.request.Request(base + "/reset", data=reset_data, headers={"Content-Type": "application/json"}, method="POST")
urllib.request.urlopen(req)

# Step with correct action
step_data = json.dumps({"task_id": "easy", "action": {"category": "newsletter", "priority": "low"}}).encode()
req2 = urllib.request.Request(base + "/step", data=step_data, headers={"Content-Type": "application/json"}, method="POST")
r = json.loads(urllib.request.urlopen(req2).read())
reward = r["reward"]
print(f"easy correct reward: {reward}  | strictly in (0,1): {0.0 < reward < 1.0}")

# Step with wrong action
req = urllib.request.Request(base + "/reset", data=reset_data, headers={"Content-Type": "application/json"}, method="POST")
urllib.request.urlopen(req)
step_data2 = json.dumps({"task_id": "easy", "action": {"category": "spam", "priority": "high"}}).encode()
req3 = urllib.request.Request(base + "/step", data=step_data2, headers={"Content-Type": "application/json"}, method="POST")
r2 = json.loads(urllib.request.urlopen(req3).read())
reward2 = r2["reward"]
print(f"easy wrong reward:   {reward2}  | strictly in (0,1): {0.0 < reward2 < 1.0}")

assert 0.0 < reward < 1.0, f"FAIL: {reward}"
assert 0.0 < reward2 < 1.0, f"FAIL: {reward2}"
print("SPACE REWARDS OK - all strictly in (0, 1)")
