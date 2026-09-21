export type Candidate = {
  name: string;
  confidence: number;
  // 候補ごとのカロリー。候補にカロリーを含まない旧APIに当たることがあるため任意扱い
  calories?: number | string;
  portion?: string;
};

export type PredictResult = {
  name: string;
  confidence: number;
  calories: number | string;
  portion: string;
  full_name: string;
  // 判定できたか。確信度が低い場合だけでなく、1位と2位が拮抗した場合も false になる
  determined: boolean;
  top3: Candidate[];
};
