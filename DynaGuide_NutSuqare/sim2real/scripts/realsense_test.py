import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print("Devices found:", len(devices))

for dev in devices:
    print("Device Name:", dev.get_info(rs.camera_info.name))
    for sensor in dev.sensors:
        print("Sensor Name:", sensor.get_info(rs.camera_info.name))