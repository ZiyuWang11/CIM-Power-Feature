// adc_ref.c - test generate an ADC energy LUT

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BIT 8
#define LENGTH 256

float* adcRef(int n);

int main() 
{
    float* adcRefArray;

    adcRefArray = adcRef(BIT);

    for (int i = 0; i < LENGTH; ++i) {
        printf("%f\n", adcRefArray[i]);
    }

    return 0;
}

float* adcRef(int n)
{
    // Create an array to be returned
    static float arr[LENGTH];

    // Initialize cap array
    int c[BIT+1];
    int sumc = 0;
    for (int i = 0; i < BIT; ++i) {
        c[i] = 1 << (BIT - i - 1);
        sumc += c[i];
    }
    // Ref cap
    c[BIT+1] = 1;
    sumc += 1;
   
    float vref = 1.0;

    // Loop through all output codes
    // i is normalized Vin
    for (int i = 0; i < LENGTH; ++i) {
        float vin = (float)i / (float)(LENGTH - 1);
        printf("Input Voltage: %f\n", vin);
        // Initialize switches, comparator voltage and energy
        int s[BIT];
        memset(s, 0, sizeof(s[0])*BIT);

        float vcomp = 0.0;
        float vcomp_prev = 0.0;
        float energy = 0.0;

        // Initial charge
        float q = - vin * sumc;
        
        // Loop through all DAC switching
        for (int j = 0; j < BIT; ++j) {
            // close switch
            s[j] = 1;
            // generate a new vcomp
            vcomp_prev = vcomp;
            float c_connect = 0.0;
            for (int k = 0; k < BIT; ++k) {
                c_connect += s[k] * c[k];
            }
            vcomp = (q + c_connect * vref) / sumc;

            if (j == 0) {
                energy += c[j] * vref * (- vin - (vcomp - vref));
            }
            else {
                float c_switch = 0.0;
                for (int k = 0; k < j; ++k) {
                    c_switch += s[k] * c[k];
                }
                energy += c_switch * vref * (vcomp_prev - vcomp);
                energy += c[j] * vref * (vcomp_prev - (vcomp - vref));
            }

            s[j] = vcomp < 0.0 ? 1 : 0;
        }
       
        arr[i] = energy;
    }

    return arr;
}
