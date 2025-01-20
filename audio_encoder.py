import torch
import torch.nn as nn

class AcousticEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_mobilenet_blocks=5):
        """
        Args:
            input_dim: 입력 오디오 특성 차원 (예: Mel-spectrogram 크기)
            hidden_dim: MobileNet 블록의 출력 차원 
            num_classes: CTC 분포를 위한 출력 클래스 수 (예: 알파벳 문자 + blank)
            num_mobilenet_blocks: MobileNet 블록의 개수
        """
        super(AcousticEncoder, self).__init__()
        
        # 스트리밍 모드를 위한 MobileNet 블록 스택
        self.mobilenet_blocks = nn.Sequential(
            *[DepthwiseSeparableConv(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=1)
              for i in range(num_mobilenet_blocks)]
        )
        
        # CTC 프로젝션 블록 - 각 프레임에 대한 토큰 분포 P(y|xt) 생성
        self.ctc_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # AE 프로젝션 블록 - 프레임 수준 임베딩 hframeAE_t 생성
        self.ae_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, audio_features):
        """
        Args:
            audio_features: (Batch, Input_Dim, Time_Steps) - 스트리밍 오디오 입력 X={xt|1≤t≤T}
        Returns:
            ctc_logits: (Batch, Time_Steps, Num_Classes) - 각 프레임의 토큰 분포 P(y|xt)
            frame_embeddings: (Batch, Time_Steps, Hidden_Dim) - 프레임 수준 AE hframeAE_t
        """
        # 스트리밍 오디오를 MobileNet 블록으로 인코딩
        x = self.mobilenet_blocks(audio_features)  # (Batch, Hidden_Dim, Time_Steps)
        x = x.transpose(1, 2)  # (Batch, Time_Steps, Hidden_Dim)
        
        # 각 프레임에 대한 토큰 분포 및 AE 임베딩 생성
        ctc_logits = self.ctc_projection(x)  # P(y|xt)
        frame_embeddings = self.ae_projection(x)  # hframeAE_t
        
        return ctc_logits, frame_embeddings

    def decode(self, alignments, ctc_logits):
        """
        CTC 정렬과 로짓을 사용하여 오디오를 텍스트로 디코딩
        Args:
            alignments: (Batch, Time_Steps, Num_States) - CTC 정렬 확률
            ctc_logits: (Batch, Time_Steps, Num_Classes) - CTC 로짓
        Returns:
            decoded_text: 디코딩된 텍스트 시퀀스 리스트
        """
        batch_size = alignments.shape[0]
        decoded_text = []
        
        # 각 배치에 대해 디코딩 수행
        for b in range(batch_size):
            # 가장 높은 확률의 상태 시퀀스 찾기
            best_path = torch.argmax(alignments[b], dim=1)
            
            # CTC 로짓에서 각 타임스텝의 가장 높은 확률의 토큰 찾기
            best_tokens = torch.argmax(ctc_logits[b], dim=1)
            
            # blank가 아닌 연속된 토큰들만 선택 (중복 제거)
            prev_token = None
            current_text = []
            
            for t, (state, token) in enumerate(zip(best_path, best_tokens)):
                if token != 0 and token != prev_token:  # 0은 blank 토큰
                    current_text.append(token.item())
                prev_token = token
                
            decoded_text.append(current_text)
            
        return decoded_text
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class OptimizedAcousticEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks=5):
        super().__init__()
        self.blocks = nn.Sequential(
            *[DepthwiseSeparableConv(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=1)
              for i in range(num_blocks)]
        )
        self.ctc_projection = nn.Linear(hidden_dim, num_classes)
        self.ae_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = x.transpose(1, 2)
        ctc_logits = self.ctc_projection(x)
        frame_embeddings = self.ae_projection(x)
        return ctc_logits, frame_embeddings

def test_acoustic_encoder():
    """
    AcousticEncoder 모델을 테스트하는 함수
    """
    # 테스트 파라미터 설정
    batch_size = 2
    input_dim = 80  # Mel-spectrogram 특성 차원
    time_steps = 50  # 시간 스텝 수
    hidden_dim = 128
    num_classes = 30  # 문자 클래스 수 + blank
    
    # 테스트용 입력 데이터 생성
    test_input = torch.randn(batch_size, input_dim, time_steps)
    
    # 모델 초기화
    model = AcousticEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # 모델 추론
    ctc_logits, frame_embeddings = model(test_input)
    
    # 출력 형태 검증
    print("=== AcousticEncoder 테스트 결과 ===")
    print(f"입력 크기: {test_input.shape}")
    print(f"CTC 로짓 크기: {ctc_logits.shape}")
    print(f"프레임 임베딩 크기: {frame_embeddings.shape}")
    
    # 기대하는 출력 형태와 일치하는지 확인
    assert ctc_logits.shape == (batch_size, time_steps, num_classes), \
        "CTC 로짓의 형태가 잘못되었습니다."
    assert frame_embeddings.shape == (batch_size, time_steps, hidden_dim), \
        "프레임 임베딩의 형태가 잘못되었습니다."
    
    print("모든 테스트가 통과되었습니다!")


if __name__ == "__main__":
    test_acoustic_encoder()