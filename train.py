import torch
import torch.nn as nn
import torch.optim as optim
from audio_encoder import AudioEncoder
from text_encoder import TextEncoder
from ctc_aligner import CTCAligner

class KWSTrainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.audio_encoder = AudioEncoder().to(self.device)
        self.text_encoder = TextEncoder().to(self.device)
        self.ctc_aligner = CTCAligner().to(self.device)
        
        # 손실 함수와 옵티마이저 설정
        self.criterion = nn.CTCLoss(blank=0, reduction='mean')
        self.optimizer = optim.Adam([
            {'params': self.audio_encoder.parameters()},
            {'params': self.text_encoder.parameters()},
            {'params': self.ctc_aligner.parameters()}
        ], lr=config['learning_rate'])

    def train_step(self, audio_input, text_input, audio_lengths, text_lengths):
        self.optimizer.zero_grad()
        
        # 오디오 인코딩
        audio_features = self.audio_encoder(audio_input)
        
        # 텍스트 인코딩
        text_features = self.text_encoder(text_input)
        
        # CTC 정렬 및 예측
        logits = self.ctc_aligner(audio_features, text_features)
        
        # 손실 계산
        loss = self.criterion(logits.transpose(0, 1), text_input, 
                            audio_lengths, text_lengths)
        
        # 역전파 및 옵티마이저 스텝
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

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

if __name__ == "__main__":
    # 설정
    config = {
        'learning_rate': 0.001,
        'num_epochs': 10,
        'batch_size': 32
    }
    
    # 트레이너 초기화
    trainer = KWSTrainer(config)
    
    # 데이터 로더는 사용자가 직접 구현해야 함
    # train_loader = ...
    
    # 학습 실행
    # trainer.train(train_loader, config['num_epochs'])
