import torch
import torch.nn as nn
import torch.optim as optim
from audio_encoder import AcousticEncoder 
from text_encoder import TextEncoder
from ctc_aligner import CTCAligner
from dataloader import get_data_loader, create_dummy_data


class KWSTrainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.audio_encoder = AcousticEncoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes']
        ).to(self.device)
        
        self.text_encoder = TextEncoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        self.ctc_aligner = CTCAligner(
            num_states=config['num_states'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        # 손실 함수와 옵티마이저 설정
        self.ctc_criterion = nn.CTCLoss(blank=0, reduction='mean')
        self.multiview_criterion = nn.MSELoss()
        self.optimizer = optim.Adam([
            {'params': self.audio_encoder.parameters()},
            {'params': self.text_encoder.parameters()},
            {'params': self.ctc_aligner.parameters()}
        ], lr=config['learning_rate'])
        
        # 추론을 위한 하이퍼파라미터
        self.lambda_weight = config.get('lambda_weight', 0.5)

    def train_step(self, audio_input, text_input, audio_lengths, text_lengths):
        self.optimizer.zero_grad()
        
        # 오디오 인코딩 - CTC 로짓과 프레임 수준 AE 획득
        ctc_logits, frame_embeddings = self.audio_encoder(audio_input)
        
        # 텍스트 인코딩 - 토큰 수준 TE 획득
        token_embeddings = self.text_encoder(text_input, text_lengths)
        
        # CTC 정렬 획득
        alignments = self.ctc_aligner(ctc_logits, frame_embeddings, text_input)
        
        # CTC 손실 계산
        ctc_loss = self.ctc_criterion(
            ctc_logits.transpose(0, 1), 
            text_input,
            audio_lengths, 
            text_lengths
        )
        
        # 토큰 수준 AE 계산 (식 8)
        batch_size = audio_input.size(0)
        token_ae = []
        
        for b in range(batch_size):
            # 최적 경로에서 각 non-blank 토큰의 전이 시간 획득
            best_path = torch.argmax(alignments[b], dim=1)
            transition_times = []
            prev_state = -1
            
            for t, state in enumerate(best_path):
                if state % 2 == 1 and state != prev_state:  # non-blank 토큰
                    transition_times.append(t)
                prev_state = state
            
            # 각 토큰에 대한 프레임 수준 AE 평균 계산
            token_embeddings_b = []
            for i in range(len(transition_times)-1):
                start_t = transition_times[i]
                end_t = transition_times[i+1]
                token_ae_i = frame_embeddings[b, start_t:end_t].mean(dim=0)
                token_embeddings_b.append(token_ae_i)
                
            token_ae.append(torch.stack(token_embeddings_b))
            
        token_ae = nn.utils.rnn.pad_sequence(token_ae, batch_first=True)
        
        # Multi-view 손실 계산 (식 9)
        multiview_loss = self.multiview_criterion(token_ae, token_embeddings)
        total_loss = ctc_loss + multiview_loss
        
        # 역전파 및 옵티마이저 스텝
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (audio, text, audio_lens, text_lens) in enumerate(train_loader):
                audio = audio.to(self.device)
                text = text.to(self.device)
                audio_lens = audio_lens.to(self.device)
                text_lens = text_lens.to(self.device)
                
                loss = self.train_step(audio, text, audio_lens, text_lens)
                total_loss += loss
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss:.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}')
            
def inference(model, audio_input, keyword_text, audio_lengths, text_lengths):
    """
    오디오와 키워드에 대한 검출 점수 계산
    Args:
        model: KWSTrainer 모델 인스턴스
        audio_input: 입력 오디오 특징
        keyword_text: 검색할 키워드 텍스트
        audio_lengths: 오디오 길이
        text_lengths: 텍스트 길이
    Returns:
        final_scores: 각 배치에 대한 최종 검출 점수
    """
    # 오디오 인코딩
    ctc_logits, frame_embeddings = model.audio_encoder(audio_input)
    
    # 키워드 텍스트 인코딩
    token_embeddings = model.text_encoder(keyword_text, text_lengths)
    
    # CTC 정렬 획득
    alignments = model.ctc_aligner(ctc_logits, frame_embeddings, keyword_text)
    
    batch_size = audio_input.size(0)
    final_scores = []
    
    for b in range(batch_size):
        # CTC 점수 - 마지막 상태의 점수 사용
        ctc_score = alignments[b, -1].max()
        
        # 임베딩 점수 계산
        best_path = torch.argmax(alignments[b], dim=1)
        transition_times = []
        prev_state = -1
        
        # non-blank 토큰의 전이 시간 찾기
        for t, state in enumerate(best_path):
            if state % 2 == 1 and state != prev_state:
                transition_times.append(t)
            prev_state = state
        
        # 토큰 수준 AE 계산
        token_ae = []
        for i in range(len(transition_times)-1):
            start_t = transition_times[i]
            end_t = transition_times[i+1]
            token_ae_i = frame_embeddings[b, start_t:end_t].mean(dim=0)
            token_ae.append(token_ae_i)
        
        token_ae = torch.stack(token_ae)
        
        # 코사인 유사도 계산
        cos_sim = nn.functional.cosine_similarity(
            token_ae.unsqueeze(1),
            token_embeddings[b, :text_lengths[b]].unsqueeze(0),
            dim=2
        ).mean()
        
        # 최종 점수 계산
        final_score = ctc_score + model.lambda_weight * cos_sim
        final_scores.append(final_score.item())
        
    return final_scores

def test_inference():
    """
    Inference 함수를 테스트하는 함수
    """
    # 테스트 파라미터 설정
    config = {
        'input_dim': 80,
        'hidden_dim': 128,
        'num_classes': 30,
        'num_states': 15,
        'learning_rate': 0.001,
        'lambda_weight': 0.5
    }
    
    # 모델 초기화
    model = KWSTrainer(config)
    
    # 테스트 데이터 생성
    batch_size = 2
    time_steps = 50
    max_text_len = 5
    
    audio_input = torch.randn(batch_size, config['input_dim'], time_steps)
    keyword_text = torch.randint(0, config['num_classes'], (batch_size, max_text_len))
    audio_lengths = torch.tensor([time_steps, time_steps-10])
    text_lengths = torch.tensor([max_text_len, max_text_len-1])
    
    # Inference 실행
    scores = inference(model, audio_input, keyword_text, audio_lengths, text_lengths)
    
    # 결과 검증
    print("=== Inference 테스트 결과 ===")
    print(f"입력 오디오 크기: {audio_input.shape}")
    print(f"키워드 텍스트 크기: {keyword_text.shape}")
    print(f"검출 점수: {scores}")
    
    # 기본적인 검증
    assert len(scores) == batch_size, "배치 크기가 일치하지 않습니다"
    assert all(0 <= score <= 2 for score in scores), "점수가 예상 범위를 벗어났습니다"
    
    print("모든 테스트가 통과되었습니다!")


if __name__ == "__main__":
    # 설정
    config = {
        'input_dim': 80,  # 멜스펙트로그램 특성 차원
        'hidden_dim': 128,
        'num_classes': 30,  # 문자 클래스 수 + blank
        'num_states': 15,  # 2U-1 (U는 최대 키워드 길이)
        'learning_rate': 0.001,
        'num_epochs': 10,
        'batch_size': 32,
        'lambda_weight': 0.5  # 임베딩 점수 가중치
    }
    
    # 트레이너 초기화
    trainer = KWSTrainer(config)
    
    # 더미 데이터 생성
    audio_features, text_labels = create_dummy_data(1000)
    
    # 데이터 로더 생성
    train_loader = get_data_loader(
        audio_features,
        text_labels,
        batch_size=config['batch_size']
    )
    
    # 학습 실행
    trainer.train(train_loader, config['num_epochs'])
