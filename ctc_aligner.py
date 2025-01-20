import torch
import torch.nn as nn


class CTCAligner(nn.Module):
    def __init__(self, num_states, hidden_dim):
        """
        Args:
            num_states: 디코딩 그래프의 상태 수 (2U - 1)
            hidden_dim: AE 및 CTC 출력 차원
        """
        super(CTCAligner, self).__init__()
        self.num_states = num_states
        self.hidden_dim = hidden_dim
        
        # CTC 정렬을 위한 추가 레이어
        self.alignment_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)
        )

    def forward(self, ctc_logits, frame_embeddings, keyword_tokens):
        """
        Args:
            ctc_logits: (Batch, Time_Steps, Num_Classes) - CTC 분포
            frame_embeddings: (Batch, Time_Steps, Hidden_Dim) - 프레임 수준 AE
            keyword_tokens: 키워드 토큰 (Padding 포함)
        Returns:
            alignments: CTC 정렬 결과 (Batch, Time_Steps, Num_States)
        """
        batch_size, time_steps, _ = ctc_logits.shape
        
        # CTC 로그 확률 계산
        log_probs = torch.log_softmax(ctc_logits, dim=-1)
        
        # Forward 알고리즘
        forward_vars = torch.zeros(batch_size, time_steps, self.num_states).to(ctc_logits.device)
        forward_vars[:, 0, 0] = log_probs[:, 0, keyword_tokens[:, 0]]
        
        for t in range(1, time_steps):
            # 이전 상태에서의 전이
            prev_forward = forward_vars[:, t-1].unsqueeze(2)
            
            # 현재 프레임의 방출 확률
            emit_score = log_probs[:, t].unsqueeze(1)
            
            # 상태 전이 점수 계산
            transition_scores = self.alignment_network(
                torch.cat([frame_embeddings[:, t], frame_embeddings[:, t-1]], dim=-1)
            )
            
            # Forward 변수 업데이트
            forward_vars[:, t] = torch.logsumexp(
                prev_forward + emit_score + transition_scores.unsqueeze(1),
                dim=1
            )
        
        # Backward 알고리즘
        backward_vars = torch.zeros_like(forward_vars)
        backward_vars[:, -1, -1] = 0
        
        for t in range(time_steps-2, -1, -1):
            next_backward = backward_vars[:, t+1].unsqueeze(1)
            emit_score = log_probs[:, t+1].unsqueeze(2)
            
            transition_scores = self.alignment_network(
                torch.cat([frame_embeddings[:, t+1], frame_embeddings[:, t]], dim=-1)
            )
            
            backward_vars[:, t] = torch.logsumexp(
                next_backward + emit_score + transition_scores.unsqueeze(2),
                dim=2
            )
        
        # 최종 정렬 확률 계산
        alignments = forward_vars + backward_vars
        alignments = torch.softmax(alignments, dim=-1)
        
        return alignments

    def decode(self, alignments, ctc_logits):
        """
        CTC 정렬과 로짓을 사용하여 디코딩
        Args:
            alignments: (Batch, Time_Steps, Num_States) - CTC 정렬 확률
            ctc_logits: (Batch, Time_Steps, Num_Classes) - CTC 로짓
        Returns:
            decoded_text: 디코딩된 텍스트 리스트
        """
        batch_size = alignments.shape[0]
        decoded_text = []
        
        # 각 배치에 대해 디코딩 수행
        for b in range(batch_size):
            # 가장 높은 확률의 상태 시퀀스 찾기
            best_path = torch.argmax(alignments[b], dim=1)
            
            # CTC 로짓에서 각 타임스텝의 가장 높은 확률의 토큰 찾기
            best_tokens = torch.argmax(ctc_logits[b], dim=1)
            
            # blank가 아닌 연속된 토큰들만 선택
            prev_token = None
            current_text = []
            
            for t, (state, token) in enumerate(zip(best_path, best_tokens)):
                if token != 0 and token != prev_token:  # 0은 blank 토큰
                    current_text.append(token.item())
                prev_token = token
                
            decoded_text.append(current_text)
            
        return decoded_text


def test_ctc_aligner():
    """
    CTCAligner 모델을 테스트하는 함수
    """
    # 테스트 파라미터 설정
    batch_size = 2
    time_steps = 50
    num_classes = 30
    hidden_dim = 128
    num_states = 15  # 예시 값 (실제로는 2U-1)
    
    # 테스트용 입력 데이터 생성
    ctc_logits = torch.randn(batch_size, time_steps, num_classes)
    frame_embeddings = torch.randn(batch_size, time_steps, hidden_dim)
    keyword_tokens = torch.randint(0, num_classes, (batch_size, 5))  # 5는 키워드 최대 길이
    
    # 모델 초기화
    model = CTCAligner(num_states=num_states, hidden_dim=hidden_dim)
    
    # 모델 추론
    alignments = model(ctc_logits, frame_embeddings, keyword_tokens)
    
    # 디코딩 테스트
    decoded_text = model.decode(alignments, ctc_logits)
    
    # 출력 형태 검증
    print("=== CTCAligner 테스트 결과 ===")
    print(f"CTC 로짓 크기: {ctc_logits.shape}")
    print(f"프레임 임베딩 크기: {frame_embeddings.shape}")
    print(f"키워드 토큰 크기: {keyword_tokens.shape}")
    print(f"정렬 결과 크기: {alignments.shape}")
    print(f"디코딩 결과: {decoded_text}")
    
    # 기대하는 출력 형태와 일치하는지 확인
    assert alignments.shape == (batch_size, time_steps, num_states), \
        "정렬 결과의 형태가 잘못되었습니다."
    
    # 정렬 확률의 합이 1인지 확인 (각 타임스텝에서)
    prob_sums = alignments.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums)), \
        "정렬 확률의 합이 1이 아닙니다."
    
    print("모든 테스트가 통과되었습니다!")


if __name__ == "__main__":
    test_ctc_aligner()
