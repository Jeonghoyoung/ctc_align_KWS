import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KWSDataset(Dataset):
    def __init__(self, audio_features, text_labels):
        """
        Args:
            audio_features: 오디오 특징 리스트 (numpy array 형태)
            text_labels: 텍스트 레이블 리스트
        """
        self.audio_features = audio_features
        self.text_labels = text_labels
        
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        audio = torch.FloatTensor(self.audio_features[idx])
        text = torch.LongTensor(self.text_labels[idx])
        audio_length = torch.LongTensor([audio.shape[1]])
        text_length = torch.LongTensor([len(text)])
        
        return audio, text, audio_length, text_length

def collate_fn(batch):
    """
    배치 내의 시퀀스들을 패딩하여 동일한 길이로 맞춤
    """
    # 배치에서 데이터 추출
    audio, text, audio_lengths, text_lengths = zip(*batch)
    
    # 오디오 특징 패딩
    max_audio_len = max(x.shape[1] for x in audio)
    audio_dim = audio[0].shape[0]
    padded_audio = torch.zeros(len(audio), audio_dim, max_audio_len)
    for i, audio_i in enumerate(audio):
        padded_audio[i, :, :audio_i.shape[1]] = audio_i
    
    # 텍스트 레이블 패딩
    max_text_len = max(len(x) for x in text)
    padded_text = torch.zeros(len(text), max_text_len).long()
    for i, text_i in enumerate(text):
        padded_text[i, :len(text_i)] = text_i
    
    # 길이 정보를 텐서로 변환
    audio_lengths = torch.cat(audio_lengths)
    text_lengths = torch.cat(text_lengths)
    
    return padded_audio, padded_text, audio_lengths, text_lengths

def get_data_loader(audio_features, text_labels, batch_size, shuffle=True):
    """
    데이터 로더 생성 함수
    
    Args:
        audio_features: 오디오 특징 리스트
        text_labels: 텍스트 레이블 리스트
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
    """
    dataset = KWSDataset(audio_features, text_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    return loader

def create_dummy_data(num_samples=1000):
    """
    테스트용 더미 데이터 생성
    """
    # 오디오 특징 생성 (멜 스펙트로그램 모방)
    audio_features = []
    for _ in range(num_samples):
        length = np.random.randint(100, 200)
        feature = np.random.randn(80, length)  # (멜밴드 수, 시간 스텝)
        audio_features.append(feature)
    
    # 텍스트 레이블 생성 (숫자로 인코딩된 문자열)
    text_labels = []
    for _ in range(num_samples):
        length = np.random.randint(5, 15)
        label = np.random.randint(1, 30, size=length)  # 1-29 사이의 문자 인덱스
        text_labels.append(label)
    
    return audio_features, text_labels

if __name__ == "__main__":
    # 더미 데이터 생성 및 데이터 로더 테스트
    audio_features, text_labels = create_dummy_data(1000)
    
    # 데이터 로더 생성
    train_loader = get_data_loader(
        audio_features,
        text_labels,
        batch_size=32
    )
    
    # 데이터 로더 테스트
    for batch_idx, (audio, text, audio_lens, text_lens) in enumerate(train_loader):
        print(f"배치 {batch_idx + 1}")
        print(f"오디오 특징 크기: {audio.shape}")
        print(f"텍스트 레이블 크기: {text.shape}")
        print(f"오디오 길이: {audio_lens}")
        print(f"텍스트 길이: {text_lens}")
        
        if batch_idx == 0:  # 첫 번째 배치만 출력
            break