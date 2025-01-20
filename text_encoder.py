import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        """
        Args:
            input_dim: 각 토큰(문자)의 임베딩 차원 (예: 1-hot 또는 임베딩 벡터 크기)
            hidden_dim: Bi-LSTM의 숨겨진 상태 크기
            num_layers: Bi-LSTM의 층 수 (기본값: 2)
        """
        super(TextEncoder, self).__init__()
        
        # 문자 토큰을 임베딩 벡터로 변환하는 임베딩 레이어
        self.embedding = nn.Embedding(num_embeddings=128, embedding_dim=input_dim)
        
        # 2개의 양방향 LSTM 레이어로 구성된 인코더
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        self.hidden_dim = hidden_dim

    def forward(self, tokenized_text, text_lengths):
        """
        Args:
            tokenized_text: (Batch, Max_Seq_Length) 형태의 텍스트 토큰 텐서
            text_lengths: 각 텍스트의 실제 길이를 나타내는 텐서 (blank 토큰 제외)
        Returns:
            token_embeddings: (Batch, U, Hidden_Dim * 2) - non-blank 토큰의 Bi-LSTM 출력
                            U는 non-blank 토큰의 총 개수
        """
        # 1. 문자 토큰을 임베딩 벡터로 변환
        embedded = self.embedding(tokenized_text)  # (Batch, Max_Seq_Length, Input_Dim)
        
        # 2. 가변 길이 시퀀스를 효율적으로 처리하기 위해 PackedSequence 사용
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 3. Bi-LSTM을 통과시켜 문맥을 고려한 토큰 표현 획득
        packed_output, _ = self.bilstm(packed_input)
        
        # 4. PackedSequence를 다시 일반 텐서로 변환
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=True
        )  # (Batch, Max_Seq_Length, Hidden_Dim * 2)
        
        # 5. text_lengths를 사용하여 각 배치에서 실제 non-blank 토큰만 선택
        batch_size = output.size(0)
        token_embeddings = []
        
        for i in range(batch_size):
            # 각 배치에서 유효한 토큰만 선택 (non-blank)
            valid_tokens = output[i, :text_lengths[i]]
            token_embeddings.append(valid_tokens)
            
        # 6. 배치 내 시퀀스들을 패딩하여 하나의 텐서로 만듦
        token_embeddings = nn.utils.rnn.pad_sequence(
            token_embeddings, 
            batch_first=True
        )  # (Batch, U, Hidden_Dim * 2)
        
        return token_embeddings

    def decode(self, alignments, ctc_logits):
        """
        CTC 정렬과 로짓을 사용하여 텍스트 디코딩
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

def test_text_encoder():
    """
    TextEncoder 모델을 테스트하는 함수
    """
    # 테스트 파라미터 설정
    batch_size = 2
    max_seq_length = 10
    input_dim = 64
    hidden_dim = 128
    
    # 테스트용 입력 데이터 생성
    tokenized_text = torch.randint(0, 128, (batch_size, max_seq_length))
    text_lengths = torch.tensor([8, 6])  # 각 배치의 실제 텍스트 길이
    
    # 모델 초기화
    model = TextEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    )
    
    # 모델 추론
    token_embeddings = model(tokenized_text, text_lengths)
    
    # 출력 형태 검증
    print("=== TextEncoder 테스트 결과 ===")
    print(f"입력 텍스트 크기: {tokenized_text.shape}")
    print(f"텍스트 길이: {text_lengths}")
    print(f"토큰 임베딩 크기: {token_embeddings.shape}")
    
    # 기대하는 출력 형태와 일치하는지 확인
    max_length = text_lengths.max().item()
    assert token_embeddings.shape == (batch_size, max_length, hidden_dim * 2), \
        "토큰 임베딩의 형태가 잘못되었습니다."
    
    print("모든 테스트가 통과되었습니다!")

if __name__ == "__main__":
    test_text_encoder()