from django.shortcuts import get_object_or_404, render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from .models import InterviewAnalysis, QuestionLists
import os
from django.conf import settings
import logging
import requests
from google.cloud import speech
from google.cloud.speech import RecognitionConfig, RecognitionAudio
from google.oauth2 import service_account
from django.conf import settings
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize
import difflib
import parselmouth
import numpy as np
import base64
import io
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

credentials = service_account.Credentials.from_service_account_file(
    os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
)

logger = logging.getLogger(__name__)

class ResponseAPIView(APIView):
    parser_classes = [JSONParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, question_list_id):
        question_list = get_object_or_404(QuestionLists, id=question_list_id)
        interview_response = InterviewAnalysis(question_list=question_list)

        
        # 로그인한 사용자를 user 필드에 할당
        interview_response.user = request.user

        
        base_dir = settings.BASE_DIR
        redundant_expressions_path = os.path.join(base_dir, 'InterviewAnalyze', 'redundant_expressions.txt')
        inappropriate_terms_path = os.path.join(base_dir, 'InterviewAnalyze', 'inappropriate_terms.txt')

        try:
            with open(redundant_expressions_path, 'r') as file:
                redundant_expressions = file.read().splitlines()
            with open(inappropriate_terms_path, 'r') as file:
                inappropriate_terms = dict(line.strip().split(':') for line in file if ':' in line)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return Response({"error": "Required file not found"}, status=500)

        response_data = []
        all_responses = ""
        for i in range(1, 11):
            script_key = f'script_{i}'
            response_key = f'response_{i}'
            question_key = f'question_{i}'
            script_text = request.data.get(script_key, "")
            question_text = getattr(question_list, question_key, "")

            # 잉여 표현과 부적절한 표현을 분석
            found_redundant = [expr for expr in redundant_expressions if expr in script_text]
            corrections = {}
            corrected_text = script_text
            for term, replacement in inappropriate_terms.items():
                if term in script_text:
                    corrections[term] = replacement
                    corrected_text = corrected_text.replace(term, replacement)

            setattr(interview_response, f'response_{i}', script_text)
            setattr(interview_response, f'redundancies_{i}', ', '.join(found_redundant))
            setattr(interview_response, f'inappropriateness_{i}', ', '.join(corrections.keys()))
            setattr(interview_response, f'corrections_{i}', str(corrections))
            setattr(interview_response, f'corrected_response_{i}', corrected_text)

            response_data.append({
                'question': question_text,
                'response': script_text,
                'redundancies': found_redundant,
                'inappropriateness': list(corrections.keys()),
                'corrections': corrections,
                'corrected_response': corrected_text,
            })

            if script_text:
                all_responses += f"{script_text}\n"

        interview_response.save()

        prompt = f"다음은 사용자의 면접 응답입니다:\n{all_responses}\n\n응답이 직무연관성, 문제해결력, 의사소통능력, 성장가능성, 인성과 관련하여 적절했는지 300자 내외로 총평을 작성해줘."
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={"model": "gpt-3.5-turbo-0125", "messages": [{"role": "user", "content": prompt}]},
                timeout=10
            )
            response.raise_for_status()
            gpt_feedback = response.json().get('choices')[0].get('message').get('content')
            interview_response.overall_feedback = gpt_feedback  # 총평을 overall_feedback 필드에 저장
        except requests.exceptions.RequestException as e:
            logger.error(f"GPT API request failed: {e}")
            gpt_feedback = "총평을 가져오는 데 실패했습니다."
            interview_response.overall_feedback = gpt_feedback  # 실패 메시지를 저장

        interview_response.save()  # 변경 사항 저장

        return Response({
            'interview_id': interview_response.id,
            'responses': response_data,
            'gpt_feedback': gpt_feedback
        }, status=200)
              

        
class VoiceAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, question_list_id=None):
        if question_list_id:
            return self.handle_response_analysis(request, question_list_id)
        else:
            return self.handle_audio_analysis(request)

    def handle_response_analysis(self, request, question_list_id):
        question_list = get_object_or_404(QuestionLists, id=question_list_id)
        interview_response = InterviewAnalysis(question_list=question_list)
        
        client = speech.SpeechClient(credentials=credentials)
        config = RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            max_alternatives=2  # 2개의 대안을 요청
        )

        audio_file_path = None

        for i in range(1, 11):
            file_key = f'audio_{i}'
            if file_key not in request.FILES:
                continue

            audio_file = request.FILES[file_key]
            audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
            with open(audio_file_path, 'wb') as f:
                f.write(audio_file.read())

            audio = RecognitionAudio(content=audio_file.read())
            response = client.recognize(config=config, audio=audio)
            highest_confidence_text = ' '.join([result.alternatives[0].transcript for result in response.results])
            most_raw_text = ' '.join([result.alternatives[1].transcript for result in response.results if len(result.alternatives) > 1])

            setattr(interview_response, f'response_{i}', highest_confidence_text)

        response_data = []
        for i in range(1, 11):
            question_key = f'question_{i}'
            response_key = f'response_{i}'
            question_text = getattr(question_list, question_key, None)
            response_text = getattr(interview_response, response_key, None)

            response_data.append({
                'question': question_text,
                'response': response_text,
            })

        # 발음 분석 및 피치 분석 수행
        pronunciation_result = None
        pitch_result = None
        intensity_result = None

        if audio_file_path:
            pronunciation_result, pronunciation_message = self.analyze_pronunciation(audio_file_path, most_raw_text, highest_confidence_text)
            pitch_result, intensity_result, pitch_graph_base64, intensity_graph_base64, intensity_message, pitch_message = self.analyze_pitch(audio_file_path)

            # 분석 결과를 인터뷰 응답 객체에 저장
            interview_response.pronunciation_similarity = pronunciation_result
            interview_response.pitch_analysis = pitch_result
            interview_response.intensity_analysis = intensity_result

        interview_response.save()

        return Response({
            'interview_id': interview_response.id,
            'responses': response_data,
            'pronunciation_similarity': pronunciation_result,
            'pitch_analysis': pitch_result,
            'intensity_analysis': intensity_result,
            'intensity_message': intensity_message,
            'pitch_message': pitch_message,
            'pronunciation_message': pronunciation_message
        }, status=200)

    def handle_audio_analysis(self, request):
        try:
            # 오디오 파일 확인
            audio_files = [request.FILES.get(f'audio_{i}') for i in range(1, 11) if request.FILES.get(f'audio_{i}')]

            if not audio_files:
                return Response({"error": "Audio files not provided"}, status=400)

            # 오디오 파일들을 하나로 병합 및 모노로 변환
            combined_audio, sample_rate = self.combine_audio_files(audio_files)
            combined_audio_path = os.path.join(settings.MEDIA_ROOT, 'combined_audio.wav')
            combined_audio.export(combined_audio_path, format='wav')

            # 발음 분석 결과 가져오기
            pronunciation_result, highest_confidence_text, average_similarity, pronunciation_message = self.analyze_pronunciation(combined_audio_path, sample_rate)

            # 피치 분석 결과 가져오기
            pitch_result, intensity_result, pitch_graph_base64, intensity_graph_base64, intensity_message, pitch_message = self.analyze_pitch(combined_audio_path)

            # JSON 형식의 결과 반환
            return Response({
                "pronunciation_similarity": pronunciation_result,
                "highest_confidence_text": highest_confidence_text,
                "average_similarity": average_similarity,
                "pitch_analysis": pitch_result,
                "intensity_analysis": intensity_result,
                "pitch_graph": pitch_graph_base64,
                "intensity_graph": intensity_graph_base64,
                "intensity_message": intensity_message,
                "pitch_message": pitch_message,
                "pronunciation_message": pronunciation_message
            }, status=200)

        except Exception as e:
            logger.error(f"Error in VoiceAPIView: {str(e)}")
            return Response({"error": "Internal Server Error", "details": str(e)}, status=500)

    def combine_audio_files(self, audio_files):
        """여러 개의 오디오 파일을 하나로 병합하고 모노로 변환"""
        combined = AudioSegment.empty()
        sample_rate = None

        for audio_file in audio_files:
            audio_segment = AudioSegment.from_file(audio_file)
            audio_segment = audio_segment.set_channels(1)  # 모노로 변환

            if sample_rate is None:
                sample_rate = audio_segment.frame_rate
            elif sample_rate != audio_segment.frame_rate:
                audio_segment = audio_segment.set_frame_rate(sample_rate)  # 프레임 레이트를 통일

            combined += audio_segment
        return combined, sample_rate

    def analyze_pronunciation(self, audio_file_path, sample_rate):
        """음성 파일의 발음 분석을 수행합니다."""
        with open(audio_file_path, 'rb') as audio_file:
            audio_content = audio_file.read()

        audio = RecognitionAudio(content=audio_content)
        config = RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code='ko-KR',
            enable_automatic_punctuation=True,
            max_alternatives=2  # 2개의 대안을 요청
        )

        client = speech.SpeechClient(credentials=credentials)
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=90)

        # 첫 번째 대안은 가장 확신도가 높은 텍스트, 두 번째 대안은 가장 원시적인 텍스트로 사용
        highest_confidence_text = response.results[0].alternatives[0].transcript
        most_raw_text = response.results[0].alternatives[1].transcript if len(response.results[0].alternatives) > 1 else highest_confidence_text

        expected_sentences = re.split(r'[.!?]', most_raw_text)
        received_sentences = re.split(r'[.!?]', highest_confidence_text)

        pronunciation_result = []
        total_similarity = 0
        num_sentences = 0

        for expected_sentence, received_sentence in zip(expected_sentences, received_sentences):
            similarity = difflib.SequenceMatcher(None, expected_sentence.strip(), received_sentence.strip()).ratio()
            total_similarity += similarity
            num_sentences += 1
            highlighted_received_sentence = self.highlight_differences(expected_sentence.strip(), received_sentence.strip(), similarity)
            pronunciation_result.append({
                '실제 발음': expected_sentence.strip(),
                '기대 발음': highlighted_received_sentence,
                '유사도': similarity
            })

        average_similarity = total_similarity / num_sentences if num_sentences > 0 else 0

        # 발음 유사도에 따른 메시지
        if average_similarity >= 0.91:
            pronunciation_message = "훌륭한 발음을 보여주셨습니다. 면접관들에게 전달하고자 하는 메시지를 명확하게 전달할 수 있는 발음입니다. 이대로 계속 연습하면 좋은 결과가 있을 것입니다."
        elif 0.81 <= average_similarity < 0.91:
            pronunciation_message = "발음이 전반적으로 괜찮습니다만, 일부 단어에서 조금 더 명확하게 발음하려는 노력이 필요할 것 같습니다. 특히 긴장하거나 빠르게 말할 때 발음이 흐려질 수 있으니, 천천히 말하며 연습해 보세요."
        else:
            pronunciation_message = "발음 연습이 조금 더 필요해 보입니다. 면접관에게 전달하고자 하는 메시지를 명확하게 전달하기 위해 중요한 단어들을 뚜렷하게 발음하는 연습을 추천드립니다. 꾸준한 연습을 통해 발음을 개선해 나가면 좋겠습니다."

        return pronunciation_result, highest_confidence_text, average_similarity, pronunciation_message

    def highlight_differences(self, expected_sentence, received_sentence, similarity):
        """예상 문장과 받은 문장의 차이점을 강조합니다."""
        if similarity > 0.9:
            return received_sentence  # similarity가 0.9 이상이면 강조하지 않음

        sequence_matcher = difflib.SequenceMatcher(None, expected_sentence, received_sentence)
        highlighted_received_sentence = ""
        for opcode, a0, a1, b0, b1 in sequence_matcher.get_opcodes():
            if opcode == 'equal':
                highlighted_received_sentence += received_sentence[b0:b1]
            elif opcode == 'replace' or opcode == 'insert':
                highlighted_received_sentence += f"<span style='color:blue;'>{received_sentence[b0:b1]}</span>"
            elif opcode == 'delete':
                highlighted_received_sentence += f"<span style='color:blue;'></span>"
        return highlighted_received_sentence

    def analyze_pitch(self, audio_file_path):
        """음성 파일의 피치 분석을 수행하고 그래프를 생성합니다."""
        sound = parselmouth.Sound(audio_file_path)

        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()

        intensity = sound.to_intensity()
        intensity_values = intensity.values.T
        intensity_times = intensity.xs()

        # 0이 아닌 피치 값과 강도 값 필터링
        non_zero_pitch_values = pitch_values[pitch_values > 0]
        non_zero_intensity_values = intensity_values[intensity_values > 0]

        # 피치 그래프 생성
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(pitch_times / 60, pitch_values, 'o', markersize=2, label='강도')
        ax1.plot(pitch_times / 60, np.where((pitch_values >= 150) & (pitch_values <= 500), pitch_values, np.nan), 'o', markersize=2, color='blue', label='일반적(150-500Hz)')
        ax1.plot(pitch_times / 60, np.where((pitch_values < 150) | (pitch_values > 500), pitch_values, np.nan), 'o', markersize=2, color='red', label='범위 밖')
        ax1.set_xlabel('시간(분)')
        ax1.set_ylabel('피치(Hz)')
        ax1.set_title('Pitch')
        ax1.set_xlim([0, max(pitch_times / 60)])
        ax1.set_ylim([0, 500])
        ax1.grid(True)
        ax1.legend()
        ax1.set_xticks(np.arange(0, max(pitch_times / 60), 1))

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pitch_graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        # 강도 그래프 생성
        fig, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(intensity_times / 60, intensity_values, linewidth=1, label='강도')
        ax2.plot(intensity_times / 60, np.where((intensity_values >= 35) & (intensity_values <= 65), intensity_values, np.nan), linewidth=1, color='blue', label='일반적(35-65db)')
        ax2.plot(intensity_times / 60, np.where((intensity_values < 35) | (intensity_values > 65), intensity_values, np.nan), linewidth=1, color='red', label='범위 밖')
        ax2.set_xlabel('시간(분)')
        ax2.set_ylabel('강도(dB)')
        ax2.set_title('Intensity')
        ax2.set_xlim([0, max(intensity_times / 60)])
        ax2.set_ylim([0, max(intensity_values)])
        ax2.grid(True)
        ax2.legend()
        ax2.set_xticks(np.arange(0, max(intensity_times / 60), 1))

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        intensity_graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        pitch_result = {
            'times': (pitch_times / 60).tolist(),  # 분 단위로 변환
            'values': pitch_values.tolist(),
            'min_value': float(np.min(non_zero_pitch_values)),
            'max_value': float(np.max(non_zero_pitch_values)),
            'average_value': float(np.mean(non_zero_pitch_values))
        }

        intensity_result = {
            'times': (intensity_times / 60).tolist(),  # 분 단위로 변환
            'values': intensity_values.tolist(),
            'min_value': float(np.min(non_zero_intensity_values)),
            'max_value': float(np.max(non_zero_intensity_values)),
            'average_value': float(np.mean(non_zero_intensity_values))
        }

        # 강도 평균값 평가
        intensity_avg = intensity_result['average_value']
        if intensity_avg >= 35 and intensity_avg <= 65:
            intensity_message = "목소리 크기가 적당합니다. 면접관이 듣기 좋은 수준의 목소리를 가지고 계십니다. 이 크기로 계속 연습하시면 좋을 것입니다."
        elif intensity_avg < 35:
            intensity_message = "목소리가 다소 작은 편입니다. 면접관에게 자신감 있는 모습을 보여주기 위해 조금 더 크게 말해 보세요. 목소리 크기를 키우는 연습을 통해 더욱 당당한 인상을 줄 수 있습니다."
        else:
            intensity_message = "목소리가 다소 큰 편입니다. 조금만 더 부드럽고 차분하게 말하면 좋을 것 같습니다. 면접관에게 강한 인상을 주는 것도 좋지만, 너무 큰 목소리는 오히려 부담을 줄 수 있습니다."

        # 피치 평균값 평가
        pitch_avg = pitch_result['average_value']
        if pitch_avg >= 150 and pitch_avg <= 450:
            pitch_message = "말씀하시는 속도가 적당합니다. 면접관이 이해하기 쉬운 속도로 말하고 계십니다. 이 속도로 계속 연습하시면 좋을 결과가 있을 것입니다."
        elif pitch_avg < 150:
            pitch_message = "말씀하시는 속도가 다소 느린 편입니다. 조금 더 빠르게 말하면 면접관의 집중력을 유지하는 데 도움이 될 것입니다. 적당한 속도를 유지하며 자연스럽게 말하는 연습을 추천드립니다."
        else:
            pitch_message = "말씀하시는 속도가 조금 빠른 편입니다. 천천히 말하면 면접관이 더 잘 이해할 수 있고, 자신감 있는 모습을 보일 수 있습니다. 천천히 말하는 연습을 통해 전달력을 높여 보세요."

        return pitch_result, intensity_result, pitch_graph_base64, intensity_graph_base64, intensity_message, pitch_message


