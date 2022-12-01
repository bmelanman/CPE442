import subprocess
import sys

import ffmpeg


def timer(main_dir, media_dir, num_tests):
    num_frames = int(ffmpeg.probe(media_dir)['streams'][0]['nb_frames'])

    real_time = []
    user_time = []
    sys_time = []

    for i in range(num_tests):
        output = subprocess.getoutput("time -p " + main_dir + "/main " + media_dir).split()

        if len(output) != 17:

            print("Time output error: " + ' '.join(output))
            exit(-1)

        real_time.append(float(output[12]))
        user_time.append(float(output[14]))
        sys_time.append(float(output[16]))

    print("Number of tests: %d" % num_tests)
    print("")

    print("Frame rate avg:  %.3f" % (num_frames / (sum(real_time) / num_tests)))
    print("Real time avg:   %.3f" % (sum(real_time) / num_tests))
    print("Max real time:   %.3f" % max(real_time))
    print("Min real time:   %.3f" % min(real_time))
    print("")

    print("User time avg:   %.3f" % (sum(user_time) / num_tests))
    print("Sys time avg:    %.3f" % (sum(sys_time) / num_tests))
    print("Total time avg:  %.3f" % sum(real_time))
    print("")

    print("Raw Data: ")
    print("Real Time: " + str(real_time))
    print("User Time: " + str(user_time))
    print("Syst Time: " + str(sys_time))
    print("")


if len(sys.argv) != 5:
    print("Invalid args: " + str(len(sys.argv)))
    exit(-1)

lab5_dir = sys.argv[1]
lab6_dir = sys.argv[2]
video_dir = sys.argv[3]
tests = int(sys.argv[4])

timer(lab5_dir, video_dir, tests)
timer(lab6_dir, video_dir, tests)
