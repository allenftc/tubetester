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

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef enum MotorState {
    IDLE,
    OPENING,
    TURNING,
    CLOSING,
    HOMING
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
float totalEncoderValue = 0;
//int16_t ticksPerOpening = 1400;
//int16_t ticksPerClosing = 500;
float absolutePosition = 0.0f;
float degreesPerTick = 360.0f / 5820.0f;

float openPos = 75.0f;
float closePos = 100.0f;


float targetPosition = 0.0f; // Target position in degrees
float kP = 0.001f; // Proportional gain for the control loop
float encoderKP = 0.0005f; // Proportional gain for the encoder feedback
uint32_t cycleTime = 0;
float dutyCycle = 0.0f;
float offset = 35.0f; // Offset for the absolute position
int16_t lastPos = 0; // Last position for calculating speed
int16_t speed = 0; // Speed in degrees per second
MotorState motorState = IDLE; // Current state of the motor
uint8_t accelerationCounter = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void setMotorPower(float power);
float wrapAngle(float degrees);
float round(float value, float multiple);
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
  HAL_Delay(50); // 50ms is plenty for the host PC to register the disconnectP

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
    
    printf("Absolute Position: %i degrees, Encoder Value: %i, Target: %i degrees, Speed: %i deg/s\n", (int)absolutePosition, (int)(totalEncoderValue), (int)targetPosition, (int) speed);
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
        dutyCycle = (float)HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1) / cycleTime;
        absolutePosition = wrapAngle(dutyCycle * 360.0f + offset); // Convert duty cycle to degrees

        encoderValue = (int16_t)(TIM2->CNT); // Read the encoder value from TIM2 counter
        speed = (encoderValue); // Calculate speed in degrees per second
        totalEncoderValue += speed * degreesPerTick; // Update total encoder value
        totalEncoderValue = wrapAngle(totalEncoderValue); // Wrap total encoder value to -180 to 180 degrees

        switch (motorState) {
            case IDLE:
                setMotorPower(0.0f); // Ensure motor is stopped
                break;
            case OPENING:
                if (fabsf(wrapAngle(totalEncoderValue - targetPosition)) < 5.0f) { // If close to target
                    setMotorPower(0.0f); // Stop the motor
                    
                    motorState = IDLE; // Transition to idle state
                } else if (fabsf(speed) < 1.0f) {
                    setMotorPower(1.0f); //Can only go one way
                }
                else {
                  setMotorPower(0.8);
                }
                break;
            case CLOSING:
                if (fabsf(wrapAngle(totalEncoderValue - targetPosition)) < 5.0f) { // If close to target
                    setMotorPower(0.0f); // Stop the motor
                    
                    motorState = IDLE; // Transition to idle state
                } else if (fabsf(speed) < 1.0f) {
                    setMotorPower(1.0f); //Can only go one way
                }
                else {
                  setMotorPower(0.8);
                }
                break;            
            case TURNING:
                // Control the motor to reach the target position
                if (fabsf(wrapAngle(targetPosition - absolutePosition)) < 5.0f) {
                    //setMotorPower(1.0f); // Stop the motor when close to target
                    setMotorPower(0.0f); // Stop the motor when close to target
                    motorState = IDLE; // Transition back to idle state
                } else if (fabsf(speed) < 1.0f) {
                    setMotorPower(-0.6); //Can only go one way
                }
                else {
                  setMotorPower(-0.15);
                }
                break;
            case HOMING:
                if (fabsf(speed) < 1.0f && accelerationCounter > 50) { // If speed is low and enough time has passed
                    setMotorPower(0.0f);
                    totalEncoderValue = absolutePosition; // Reset total encoder value to current absolute position
                    TIM2->CNT = 0; // Reset encoder count
                    accelerationCounter = 0; // Reset acceleration counter
                    motorState = IDLE; // Transition back to idle state
                }
                else if (fabsf(speed) < 1.0f) {
                    setMotorPower(-0.7); //Can only go one way
                    accelerationCounter++;
                }
                else {
                  setMotorPower(-0.3);
                  accelerationCounter = 0; // Reset acceleration counter if speed is not low
                }
        }
        TIM2->CNT = 0; // Reset encoder count for next measurement
    }
}

float wrapAngle(float degrees) {
	while (degrees > 180.0f) degrees -= 360.0f;
  while (degrees < -180.0f) degrees += 360.0f;
  return degrees;
}

float round(float value, float multiple) {
    return roundf(value / multiple) * multiple;
}


void USB_CDC_RxHandler(uint8_t* Buf, uint32_t Len)
{
	char cmd = Buf[0];
  int16_t value = atoi((char*)&Buf[1]); // Convert the rest of the buffer to an integer
  switch (cmd) {
    case 'O': // Open command
        motorState = OPENING;
        TIM2->CNT = 0; // Reset encoder count
        //targetPosition = round(absolutePosition, 90.0f) + openPos; // Set target position for opening
        targetPosition = wrapAngle(absolutePosition + openPos);
        printf("Opening command received. Encoder target: %i ticks\n", (int) targetPosition);
        break;
    case 'T': // Turn command
        targetPosition = (float)value; // Set target position in degrees
        motorState = TURNING;
        printf("Turn command received. Target position: %.2f degrees\n", targetPosition);
        break;
    case 'S': // Stop command
        setMotorPower(0.0f);
        motorState = IDLE; // Stop the motor
        printf("Stop command received. Motor stopped.\n");
        break;
    case 'C': // Close command
        motorState = CLOSING;
        TIM2->CNT = 0; // Reset encoder count
        targetPosition = wrapAngle(absolutePosition + closePos); // Set target position for closing
        printf("Closing command received. Encoder target: %i ticks\n", (int) targetPosition);
        break;
    case 'H': // Home command
        motorState = HOMING;
        printf("Home command received. Starting homing sequence.\n");
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
