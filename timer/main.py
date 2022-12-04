import re
import subprocess
import sys
import time


def timer(main_dir, media_dir, num_tests, max_execution_time=30):
    real_time = []
    user_time = []
    sys_time = []
    num_frames = 0
    skip_count = 0

    for _ in range(num_tests):
        run_time = time.time()

        # Run the code to be tested
        process = subprocess.Popen("time -p " + main_dir + "main " + media_dir, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # Wait for the program to finish
        while process.poll() is None:
            # Check if the program is taking too long
            if (time.time() - run_time) > max_execution_time:
                # Kill 'time'
                process.kill()
                # Kill 'main'
                subprocess.run("kill $(pidof main)")
                # Inform the user
                skip_count = skip_count + 1
                print("Number of skipped tests: " + str(skip_count))
                # Continue to the next test
                break

        # Get frame rate data
        video_data = re.findall(r'\d+', str(process.communicate()[0]))

        # Get time data
        output = re.findall(r'\d*[.]\d+', str(process.communicate()[1]))

        # Make sure video data was properly collected
        if len(video_data) != 5:
            print("Video data error: " + ' '.join(video_data))
            continue

        # Make sure time data was collected
        if len(output) != 3:
            print("Time data error: " + str(output))
            exit(-1)

        num_frames = int(video_data[2])

        real_time.append(float(output[0]))
        user_time.append(float(output[1]))
        sys_time.append(float(output[2]))

    if len(real_time) <= 0 or len(user_time) <= 0 or len(sys_time) <= 0:
        return

    print("Number of tests: %d" % (num_tests - skip_count))
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

    print("\n%%%%%%%%%% Sobel Filter Data Collection %%%%%%%%%%\n")

    timer(lab5_dir, video_dir, tests)
    timer(lab6_dir, video_dir, tests)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

if __name__ == '__main__':
    main()
