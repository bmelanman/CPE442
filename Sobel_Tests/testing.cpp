//
// Created by Bryce Melander on 11/17/22.
//

#import <iostream>

using namespace std;

void function_A(int test) {

    for (int i = 0; i < 1000;) {

        if (test % 5 == 0) {
            test += 6;
        } else {
            test -= 1;
        }

        i++;
    }

    cout << test << endl;
}

void function_B(int test) {

    int i = 0;

    while (i < 1000) {

        if (test % 5 == 0) {
            test += 6;
        } else {
            test -= 1;
        }

        i++;
    }

    cout << test << endl;
}

int main() {

    function_A(0);

    function_B(0);

    return 0;
}
