import re
import subprocess
import sys

import sudo
import keyboard


def timer(main_dir, media_dir, num_tests):
    real_time = []
    user_time = []
    sys_time = []
    num_frames = 0

    for _ in range(num_tests):
        # loop_continue = False

        # Run the code to be tested
        process = subprocess.Popen("time -p " + main_dir + "main " + media_dir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # While it is running, check for stop signal
        while process.poll() is None:
            pass
        #     if keyboard.is_pressed(" "):
        #         keyboard.press("ESC")
        #         process.kill()
        #         loop_continue = True
        #
        # while keyboard.is_pressed(" "):
        #     pass
        #
        # if loop_continue:
        #     continue

        num_frames = str(process.communicate()[0]).partition("frames: ")
        output = re.findall(r'\d*[.]\d+', str(process.communicate()[1]))

        print(num_frames)
        num_frames = 1

        if len(output) != 3:
            print("Time output error: " + str(output))
            exit(-1)

        real_time.append(float(output[0]))
        user_time.append(float(output[1]))
        sys_time.append(float(output[2]))

    if len(real_time) <= 0 or len(user_time) <= 0 or len(sys_time) <= 0:
        return

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


def main():
    if len(sys.argv) != 5:
        print("Invalid args: " + str(len(sys.argv)))
        exit(-1)

    lab5_dir = sys.argv[1]
    lab6_dir = sys.argv[2]
    video_dir = sys.argv[3]
    tests = int(sys.argv[4])

    timer(lab5_dir, video_dir, tests)
    timer(lab6_dir, video_dir, tests)


if __name__ == '__main__':
    main()
