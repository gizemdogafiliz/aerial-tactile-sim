"""
Phase 2: Test contact sensor on tilthex end-effector.
Run INSIDE Docker container (tk3lab).

Steps:
  1. Start Docker: ~/tk3lab/releases/r-1.3/scripts/tk3lab-run
  2. In Docker terminal, start Gazebo with modified world:
     gz sim -r /shared-workspace/../aerial-tactile-sim/gazebo/worlds/hexa-fa-wall-world.world
  3. Run this script in another Docker terminal:
     python3 /shared-workspace/../aerial-tactile-sim/scripts/05_test_contact_sensor.py

Note: This script listens to the gz contact topic and prints
contact data when the EE tip touches the wall.
"""

import subprocess
import json
import time

def list_topics():
    """List all gz topics to find the contact sensor topic."""
    result = subprocess.run(
        ['gz', 'topic', '-l'],
        capture_output=True, text=True, timeout=5
    )
    print("Available topics:")
    for line in result.stdout.strip().split('\n'):
        print(f"  {line}")
        if 'contact' in line.lower():
            print(f"  ^^^ CONTACT TOPIC FOUND")
    return result.stdout.strip().split('\n')


def echo_topic(topic, duration=10):
    """Echo a gz topic for a given duration."""
    print(f"\nListening to {topic} for {duration}s...")
    print("Fly the hexarotor into the wall to see contact data.\n")
    try:
        proc = subprocess.Popen(
            ['gz', 'topic', '-e', '-t', topic],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        t0 = time.time()
        while time.time() - t0 < duration:
            line = proc.stdout.readline()
            if line:
                print(line.rstrip())
        proc.terminate()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    print("=== Phase 2: Contact Sensor Test ===\n")

    topics = list_topics()

    contact_topics = [t for t in topics if 'contact' in t.lower()]

    if contact_topics:
        echo_topic(contact_topics[0], duration=30)
    else:
        print("\nNo contact topic found.")
        print("Possible reasons:")
        print("  - Using original world file (need modified version)")
        print("  - Gazebo not running")
        print("  - Contact plugin not loaded")
        print("\nTry manually: gz topic -l")
