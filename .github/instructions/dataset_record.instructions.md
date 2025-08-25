---
applyTo: '**'
---

The dataset contains the following columns:

| Field Name      | Description | Column Name | Type   |
|-----------------|-------------|-------------|--------|
| Vehicle_ID      | Vehicle identification number (ascending by time of entry into section). REPEATS ARE NOT ASSOCIATED. | vehicle_id | Number |
| Frame_ID        | Frame Identification number (ascending by start time) | frame_id | Number |
| Total_Frames    | Total number of frames in which the vehicle appears in this data set | total_frames | Number |
| Global_Time     | Elapsed time in milliseconds since Jan 1, 1970. | global_time | Number |
| Local_X         | Lateral (X) coordinate of the front center of the vehicle in feet with respect to the left-most edge of the section in the direction of travel. | local_x | Number |
| Local_Y         | Longitudinal (Y) coordinate of the front center of the vehicle in feet with respect to the entry edge of the section in the direction of travel. | local_y | Number |
| Global_X        | X Coordinate of the front center of the vehicle in feet based on CA State Plane III in NAD83. | global_x | Number |
| Global_Y        | Y Coordinate of the front center of the vehicle in feet based on CA State Plane III in NAD83. | global_y | Number |
| v_length        | Length of vehicle in feet | v_length | Number |
| v_Width         | Width of vehicle in feet | v_width | Number |
| v_Class         | Vehicle type: 1 - motorcycle, 2 - auto, 3 - truck | v_class | Number |
| v_Vel           | Instantaneous velocity of vehicle in feet/second. | v_vel | Number |
| v_Acc           | Instantaneous acceleration of vehicle in feet/second². | v_acc | Number |
| Lane_ID         | Current lane position of vehicle. Lane 1 is farthest left; lane 5 farthest right; lane 6 auxiliary lane; lane 7 on-ramp; lane 8 off-ramp. | lane_id | Number |
| O_Zone          | Origin zones of the vehicles (101–111). | o_zone | Text |
| D_Zone          | Destination zones of the vehicles (201–211). | d_zone | Text |
| Int_ID          | Intersection in which the vehicle is traveling (1–4). 0 means vehicle not in an intersection. | int_id | Text |
| Section_ID      | Section in which the vehicle is traveling (0 = not in a section). | section_id | Text |
| Direction       | Moving direction: 1 - EB, 2 - NB, 3 - WB, 4 - SB. | direction | Text |
| Movement        | Movement: 1 - through, 2 - left-turn, 3 - right-turn. | movement | Text |
| Preceding       | Vehicle ID of the lead vehicle in the same lane (0 = no preceding vehicle). | preceding | Number |
| Following       | Vehicle ID of the following vehicle in the same lane (0 = no following vehicle). | following | Number |
| Space_Headway   | Space Headway in feet (distance between front-centers of vehicles). | space_headway | Number |
| Time_Headway    | Time Headway in seconds (time gap to preceding vehicle). | time_headway | Number |
| Location        | Name of street or freeway | location | Text |
