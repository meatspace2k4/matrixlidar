Listen to me. I don't know why you (your other sessions) are struggling with this so much.
You keep detecting "Unknown" when the user is CLEARLY sitting on the toilet.

Pull the most recent session from the pi at jpark@bath-pi.local
LOOK AT THE HEATMAP. NOW.

Do NOT look at any code until I tell you.
DO NOT tell me "You're right this is bugged" and then change one thing
YOU do not decide if this is working. I do. I'm the user. I'm the one using the toilet.
First screenshot - empty
Second screenshot - walking into the bathroom
Third - standing (state is correct this time)
4 - sitting on the toilet (Unknown wrong)
5 - still sitting (you can literally see my head and phone my hands are on my thighs
6 - still sitting
7 - still sitting OBVIOUSLY TAKING A SHIT
8 - standing up (before a bend down to pick up pants obviously not something you need to detect)
9 - Walking (it shows right because the toilet, as you can see, is to the left of the body and is moving in the correct direction, NOT yet exiting (the door is closed, as you can see in column H)
10 - Passing in front of the sensor on the way to sink
11 - Left of the camera, out of range (washing hands, intended to not track, still occupied as it's the only exit past the sensor)
12 - Walking right, opening the door and
13 - exited (you can see the out-swinging door is visible in its entirety through the door frame, and there is already detection for a body passing through the door frame itself, so this is also correct

Now you have NO EXCUSES for not seeing the scene. It's right in front of you, and you have the numbers on the pi you can get, and check the logs DO NOT try to parse the grid or the csv it MUST be run through 3D scene processing from the codebase (on the pi) which is the same exact version as on disk in the repo (deployed to pi)

I have been fighting with you to get this sitting detection to work. You have done everything imaginable to try patching around the issue, finally landing on (something person shaped on TOP of the toilet seat is sitting -- yeah no shit) and it still isn't working.

What are you going to do about this? You have enough samples you do not need to view the sensor in real-time to answer this. LOOK AT THE SESSIONS ALREADY RECORDED. At least several have been even recorded and labeled as an attempt to calibrate. DO NOT RECALIBRATE. Do NOT tweak the sensor's values or try to blame the scene moving -- there are references to old misaligned baseline scans, don't worry about that. The sensor has not moved. I JUST tested this. The most recent session from the pi will align with the numbered screenshots I sent you (might be off by one)

I want you to peruse this project, DO NOT talk to yourself about "what the user wants" or "I need to do this and that" I am NOT interested in excuses or rationalizing. I just want this to work. Every random change a model makes regresses some part of the detection, and the last model fixated on "wiping", trying to blame the algorithm for not being able to tell the difference, and leaving it at that. WHO CARES IF THE USER IS WIPING THEY'RE STILL ON THE TOILET. Unless they're one of those freaks who stand up to wipe, but that doesn't matter. Standing at toilet vs sitting, neither are unknown states.

This passes when all previous recorded sessions pass the stand/sit test (I DO NOT MEAN UNIT TESTS PASSING TESTS DO NOT PROVE THIS WORKS, ONLY THE HUMAN EYE DOES. What YOU can do is compare with the screenshots provided and use the MOST RECENT SESSION as your calibration.

You already have the entire scene in the data itself. AN EMPTY, UNOCCUPIED ROOM IS YOUR BASELINE.

Go. Do not talk. Code.
