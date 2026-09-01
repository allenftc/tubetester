/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "tim.h"
#include "usb_device.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include "usbd_cdc_if.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef enum MotorState {
    IDLE,
    OPENING,
    TURNING
} MotorState;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
int16_t encoderValue = 0;
int16_t ticksPerOpening = 1600;
float absolutePosition = 0.0f;


float targetPosition = 0.0f; // Target position in degrees
float kP = 0.001f; // Proportional gain for the control loop
float encoderKP = 0.0005f; // Proportional gain for the encoder feedback
uint32_t cycleTime = 0;
float dutyCycle = 0.0f;
float offset = 35.0f; // Offset for the absolute position
float lastPos = 0.0f; // Last position for calculating speed
float speed = 0.0f; // Filtered speed in degrees per second
MotorState motorState = IDLE; // Current state of the motor

/*
 * TURNING profile calibration.  The mechanism only travels in the negative
 * motor-power direction, so it must begin reducing power before the target
 * instead of correcting an overshoot by reversing.
 *
 * Set TURN_FULL_SPEED_COAST_DEGREES to the measured coast distance at
 * TURN_REFERENCE_SPEED_DEG_PER_SEC.  The profile scales that distance using
 * the measured speed and ramps power down over TURN_POWER_RAMP_DEGREES.
 */
#define TURN_MAX_POWER                 0.125f
#define TURN_MIN_POWER                 0.030f
#define TURN_STOP_TOLERANCE_DEGREES    2.0f
#define TURN_FULL_SPEED_COAST_DEGREES 60.0f
#define TURN_MIN_COAST_DEGREES         4.0f
#define TURN_REFERENCE_SPEED_DEG_PER_SEC 180.0f
#define TURN_POWER_RAMP_DEGREES       30.0f
#define TURN_POWER_SLEW_PER_UPDATE    0.008f
#define POSITION_UPDATE_PERIOD_SEC     0.010f

float turnCommandPower = 0.0f;
float turnRemainingDegrees = 0.0f;
float turnCoastDistanceDegrees = 0.0f;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void setMotorPower(float power);
float wrapAngle(float degrees);
float forwardAngle(float degrees);
void updateTurnProfile(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* Force USB Re-Enumeration */
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  // 1. Enable GPIOA Clock
  __HAL_RCC_GPIOA_CLK_ENABLE();

  // 2. Configure PA12 (USB D+) as Output Push-Pull
  GPIO_InitStruct.Pin = GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  // 3. Drive the line LOW to trick the PC into thinking the cable was unplugged
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12, GPIO_PIN_RESET);
  HAL_Delay(50); // 50ms is plenty for the host PC to register the disconnect

  // 4. De-initialize the GPIO so the native USB hardware can take it over
  HAL_GPIO_DeInit(GPIOA, GPIO_PIN_12);
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM2_Init();
  MX_TIM1_Init();
  MX_USB_DEVICE_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1); // Start PWM on TIM1 Channel 1
  HAL_TIM_Base_Start_IT(&htim3); // Start TIM3 in interrupt mode for control loop
  //HAL_ADCEx_Calibration_Start(&hadc1); // Start ADC calibration
  //HAL_ADC_Start_DMA(&hadc1, (uint32_t*)&ADCValue, 1); // Start ADC in DMA mode
  HAL_TIM_IC_Start_IT(&htim3, TIM_CHANNEL_2); // Start Input Capture on TIM3 Channel 1
  HAL_TIM_IC_Start(&htim3, TIM_CHANNEL_1); // Start Input Capture
  HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL); // Start Encoder Interface on TIM2
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    
    printf("Absolute Position: %i degrees, Encoder Value: %i, Target: %i degrees, Speed: %i deg/s\n", (int)absolutePosition, encoderValue, (int)targetPosition, (int) speed);
    HAL_Delay(10); // Delay for 0.1 second
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USB;
  PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_PLL_DIV1_5;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
void setMotorPower(float power) {
    power *= -1.0f; // Invert power to match motor direction
    if (power > 1.0f) power = 1.0f;
    if (power < -1.0f) power = -1.0f;

    if (power > 0.0f) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13, GPIO_PIN_SET);
    } 
    else if (power < 0.0f) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_SET);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13, GPIO_PIN_RESET);
    } 
    else {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_SET);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13, GPIO_PIN_SET);
    }

    uint32_t arr = __HAL_TIM_GET_AUTORELOAD(&htim1);
    uint32_t pwm = (uint32_t)(fabsf(power) * (float)arr);
    
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, pwm);
}

int _write(int file, char *ptr, int len) {
  CDC_Transmit_FS((uint8_t*)ptr, len);
  return len;
}
int _read(int file, char *ptr, int len) {
  // Implement reading from USB CDC if needed
  return 0;
}



void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM3) {
        cycleTime = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_2);
        if (cycleTime == 0U) {
            return;
        }

        dutyCycle = (float)HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1) / cycleTime;
        absolutePosition = wrapAngle(dutyCycle * 360.0f + offset); // Convert duty cycle to degrees

        // Use the shortest angular delta so crossing -180/180 does not look like a 360 degree jump.
        float tempSpeed = fabsf(wrapAngle(absolutePosition - lastPos)) / POSITION_UPDATE_PERIOD_SEC;
        speed += 0.2f * (tempSpeed - speed); // Light filtering prevents profile jitter.
        encoderValue = (int16_t)(TIM2->CNT); // Read the encoder value from TIM2 counter

        switch (motorState) {
            case IDLE:
                if (speed < 20.0f) { // If speed is low, stop the motor
                    setMotorPower(0.0f);
                }
                setMotorPower(0.0f); // Ensure motor is stopped
                // Do nothing
                break;
            case OPENING:
                if (encoderValue > ticksPerOpening - 10) { // If close to target
                    setMotorPower(0.0f); // Stop the motor
                    
                    motorState = IDLE; // Transition to idle state
                } else {
                    setMotorPower(encoderKP * (ticksPerOpening - encoderValue)); // Use encoder feedback to control motor
                }
                break;
            case TURNING:
                updateTurnProfile();
                break;
        }
        lastPos = absolutePosition; // Update last position for speed calculation
    }
}

float wrapAngle(float degrees) {
	while (degrees > 180.0f) degrees -= 360.0f;
  while (degrees < -180.0f) degrees += 360.0f;
  return degrees;
}

/* Return the distance to the target while moving only in the allowed direction. */
float forwardAngle(float degrees) {
    while (degrees < 0.0f) degrees += 360.0f;
    while (degrees >= 360.0f) degrees -= 360.0f;
    return degrees;
}

void updateTurnProfile(void) {
    float speedRatio;
    float desiredPower;
    float powerError;

    turnRemainingDegrees = forwardAngle(targetPosition - absolutePosition);
    if (turnRemainingDegrees <= TURN_STOP_TOLERANCE_DEGREES) {
        turnCommandPower = 0.0f;
        setMotorPower(0.0f);
        motorState = IDLE;
        return;
    }

    speedRatio = speed / TURN_REFERENCE_SPEED_DEG_PER_SEC;
    if (speedRatio > 1.0f) speedRatio = 1.0f;
    if (speedRatio < 0.0f) speedRatio = 0.0f;

    // At reference speed this is 60 degrees; at lower speed it is shorter.
    turnCoastDistanceDegrees = TURN_MIN_COAST_DEGREES +
        (TURN_FULL_SPEED_COAST_DEGREES - TURN_MIN_COAST_DEGREES) * speedRatio * speedRatio;

    if (turnRemainingDegrees <= turnCoastDistanceDegrees) {
        // Stop driving and let the one-direction mechanism coast into the target.
        turnCommandPower = 0.0f;
        setMotorPower(0.0f);
        return;
    }

    desiredPower = TURN_MAX_POWER *
        ((turnRemainingDegrees - turnCoastDistanceDegrees) / TURN_POWER_RAMP_DEGREES);
    if (desiredPower > TURN_MAX_POWER) desiredPower = TURN_MAX_POWER;
    if (desiredPower < TURN_MIN_POWER) desiredPower = TURN_MIN_POWER;

    // Ramp power changes to avoid an abrupt acceleration into the coast region.
    powerError = desiredPower - turnCommandPower;
    if (powerError > TURN_POWER_SLEW_PER_UPDATE) powerError = TURN_POWER_SLEW_PER_UPDATE;
    if (powerError < -TURN_POWER_SLEW_PER_UPDATE) powerError = -TURN_POWER_SLEW_PER_UPDATE;
    turnCommandPower += powerError;

    setMotorPower(-turnCommandPower); // The only permitted travel direction.
}


void USB_CDC_RxHandler(uint8_t* Buf, uint32_t Len)
{
	char cmd = Buf[0];
  int16_t value = atoi((char*)&Buf[1]); // Convert the rest of the buffer to an integer
  switch (cmd) {
    case 'O': // Open command
        motorState = OPENING;
        TIM2->CNT = 0; // Reset encoder count
        printf("Opening command received. Encoder target: %i ticks\n", ticksPerOpening);
        break;
    case 'T': // Turn command
        targetPosition = wrapAngle((float)value); // Store the target in the sensor's angular range
        turnCommandPower = 0.0f; // Every new move starts with a controlled acceleration ramp.
        motorState = TURNING;
        printf("Turn command received. Target position: %.2f degrees\n", targetPosition);
        break;
    case 'S': // Stop command
        setMotorPower(0.0f);
        motorState = IDLE; // Stop the motor
        printf("Stop command received. Motor stopped.\n");
        break;
    default:
        // Unknown command, ignore
        printf("What are you talking about like %c ahh\n", cmd);
        break;
  }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
