import serial
import time
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import pygame
from threading import Thread, Event

stop_event = Event()

# 모델과 토크나이저 로드
def load_model():
    tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
    config = BertConfig.from_pretrained('beomi/kcbert-base', num_labels=7)
    model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', config=config)

    # 모델 파일 경로 설정
    model_path = 'C:\\Users\\ODYSSEY\\sentiment7.pth'

    # 모델 로드
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # CPU 사용
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# 추론 함수 정의
def inference(input_doc):
    inputs = tokenizer(input_doc, return_tensors='pt')
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    class_idx = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    results = {class_name: prob for class_name, prob in zip(class_idx, probs)}
    # 가장 높은 확률의 클래스를 찾음
    max_prob_class = max(results, key=results.get)
    max_prob = results[max_prob_class]
    # 결과 표시
    return [results, max_prob_class]

def main():
    print('감정분석 프로그램입니다. 글에 나타난 공포, 놀람, 분노, 슬픔, 중립, 행복, 혐오의 정도를 비율로 알려드립니다.')

    while True:
        user_input = input("이 곳에 글 입력(100자 이하 권장): ")

        result = inference(user_input)
        print('감정 분석 결과:', result[0])
        print(f"가장 강하게 나타난 감정: {result[1]}")

        # 중립 감정인 경우 재입력 요청
        if result[1] == '중립':
            print("감정이 중립으로 나타났습니다. 다른 텍스트를 입력해 주세요.")
        else:
            print(f"감정 '{result[1]}'이 분석되었습니다.")
            break

    if result[1] == '행복':
        emotion = 'happy'
    elif result[1] == '슬픔':
        emotion = 'sad'
    elif result[1] == '공포' or result[1] == '놀람':
        emotion = 'wonder'
    elif result[1] == '분노' or result[1] == '혐오':
        emotion = 'angry'
    else:
        print("감정 입력 오류")
        return

    # Arduino port
    arduino_port = 'COM5'

    # Amplitude thresholds
    set1 = 3000000
    set2 = 7000000
    set3 = 10000000
    ti_set = 2  # 2 seconds interval

    # File paths
    file_paths = {
        'happy': 'C:/Users/ODYSSEY/Desktop/file/happy - 스티커 사진.wav',
        'sad': 'C:/Users/ODYSSEY/Desktop/file/Sad - Love Story.wav',
        'angry': 'C:/Users/ODYSSEY/Desktop/file/Angry - Flowering(Cover).wav',
        'wonder': 'C:/Users/ODYSSEY/Desktop/file/wonder - 소나기 이클립스.wav'
    }

    # Fourier transform and average amplitude calculation function
    def calculate_average_amplitude_in_frequency_range(waveform, sample_rate, freq_start, freq_end):
        spectrum = fft(waveform)
        freqs = np.fft.fftfreq(len(waveform)) * sample_rate
        idx_start = np.where(freqs >= freq_start)[0][0]
        idx_end = np.where(freqs <= freq_end)[0][-1]
        avg_amplitude = np.mean(np.abs(spectrum[idx_start:idx_end]) ** 2)
        if avg_amplitude <= set1:
            pwm_value = 1
        elif set1 + 1 <= avg_amplitude <= set2:
            pwm_value = 2
        elif set2 + 1 <= avg_amplitude <= set3:
            pwm_value = 3
        else:
            pwm_value = 4
        return pwm_value

    # Audio file processing function
    def process_audio(file_path, emotion, freq_start=20, freq_end=20000):
        sample_rate, data = wavfile.read(file_path)
        ti_passed = 0.0
        result_list = []
        for i in range(0, len(data), int(sample_rate * ti_set)):
            pwm_value = calculate_average_amplitude_in_frequency_range(data[i:i + int(sample_rate * ti_set)], sample_rate, freq_start, freq_end)

            # Adjust pwm_value based on emotion
            if emotion == 'sad':  # 감정이 'sad'일 때
                pwm_value += 4
            elif emotion == 'happy':  # 감정이 'happy'일 때
                pwm_value += 8
            elif emotion == 'angry':
                pwm_value += 0  # 감정이 'angry'일 때, 추가 값 없음
            elif emotion == 'wonder':  # 감정이 'wonder'일 때
                pwm_value += 12

            result_list.append((ti_passed, pwm_value))
            ti_passed += ti_set
        watch_values, pwm_values = zip(*result_list)
        return watch_values, pwm_values, result_list

    # Serial communication and result sending function
    def send_to_arduino(result_list):
        baud_rate = 9600
        try:
            ser = serial.Serial(arduino_port, baud_rate)
            time.sleep(2)  # Wait for the serial connection to initialize
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            return

        for tm, pwm in result_list:
            if stop_event.is_set():
                break
            try:
                print(f"Time: {tm:.2f} seconds, PWM: {pwm}")
                ser.write(f"{pwm}\n".encode())
                time.sleep(ti_set)  # Ensure the timing matches the audio processing
            except serial.SerialException as e:
                print(f"Error writing to serial port: {e}")
                break
        ser.close()

    # Music playback function
    def play_music(file_path, stop_event):
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(file_path)
        except Exception as e:
            print(f"Error loading music file: {e}")
            return
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if stop_event.is_set():
                pygame.mixer.music.stop()
                break
            time.sleep(0.1)
        pygame.mixer.quit()

    # File path selection
    file_path = file_paths[emotion]

    # Start music playback in a separate thread
    music_thread = Thread(target=play_music, args=(file_path, stop_event))
    music_thread.start()

    # Process audio file
    time_values, pwm_values, result_list = process_audio(file_path, emotion)

    # Print results
    for tm, pwm in result_list:
        print(f"Time: {tm:.2f} seconds, PWM: {pwm}")

    # Send results to Arduino
    send_to_arduino(result_list)

    # Ensure the music thread finishes
    stop_event.set()
    music_thread.join()

if __name__ == "__main__":
    main()