# Optics of Intelligence / 知性光学

Code repository for the E8 series papers on Structure, Value, and Meaning in Intelligence.

E8シリーズ論文（知性における構造・価値・意味）のコードリポジトリである。

[English](#english) | [日本語](#japanese)

---

<a name="english"></a>
## English

### Papers

- **E8a** (2025-137): Observation - Discovery of VS≈0 (natural orthogonality)
- **E8b** (2025-138): Laws - Establishment of O1-O4 universal laws
- **E8c** (2025-139): Equations - Geometric formalization (V=ν⊙ψ, M=Π_V(S))
- **E8d** (2025-140): Integration - Relativity Theory of Intelligence
- **E8e** (2025-141): Verification - Code validity and reproducibility
- **E8f** (2025-142): Perfection - Complete deterministic implementation

All preprints available at [Zenodo Kakushin Structural Theory Community](https://zenodo.org/communities/kakushin-structural-theory/).

### Quick Start

#### ⚡ Option 1: Run All Experiments (Recommended for Replication)

For researchers replicating the complete E8 study:
```bash
# Clone repository
git clone https://github.com/HIDEKI-SQ/optics-of-intelligence.git
cd optics-of-intelligence

# Install dependencies
pip install -r requirements.txt

# Run all experiments (β, exp00-13)
python run_all.py
```

Results are saved to `outputs/` with hash verification in `outputs/hash_manifest.json`.

**Expected runtime:** 10-30 minutes depending on hardware.

---

#### 🎓 Option 2: Single Experiment (Google Colab)

For quick exploration or teaching purposes:

1. Navigate to [`colab/`](colab/) folder
2. Open any `exp0X_standalone.py` file
3. Copy entire contents
4. Paste into Google Colab cell
5. Click "Run" (Ctrl+Enter or Cmd+Enter)

**Example:** Try [`colab/exp01_standalone.py`](colab/exp01_standalone.py) to observe the VS≈0 phenomenon that challenged 2000 years of Method of Loci assumptions.

**No installation required** - all dependencies are standard in Google Colab.

---

#### 🔬 Option 3: Advanced Usage

For extending or modifying experiments:
```python
from src.experiments.exp_01 import run_exp01

# Run with custom seed
results = run_exp01(seed=123)

# Access specific metrics
print(f"VS (spatial): {results['vs_spatial']:.3f}")
print(f"VS (random): {results['vs_random']:.3f}")
```

See [`src/`](src/) directory for modular code structure.

---

### Key Findings

**Natural Orthogonality (O1):**
Structure and meaning are independent without value pressure (λ=0), yielding VS≈0 across all conditions.

**Phase Dominance (O2):**
Structure preservation depends on topology (φ) rather than metric geometry.

**Stress Tolerance (O3):**
Structural disruption does not automatically cause semantic confusion.

**Value-Gated Coupling (O4):**
Only value (λ>0) binds structure to meaning, achieving VS>0.

---

### System Requirements

- Python 3.11+
- numpy, scipy, pandas, matplotlib, scikit-learn
- See [`requirements.txt`](requirements.txt) for exact versions

---

### Repository Structure
```
optics-of-intelligence/
├── README.md              # This file
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies
├── run_all.py            # Execute all experiments
├── colab/                # Standalone files for Google Colab
│   ├── exp01_standalone.py
│   └── ...
├── src/                  # Modular source code
│   ├── core/             # Core functionality
│   │   ├── config.py     # Configuration
│   │   ├── measures.py   # SP/GEN/VS metrics
│   │   ├── utils.py      # Utilities
│   │   └── hash_recorder.py
│   └── experiments/      # Individual experiments
│       ├── exp_beta.py
│       ├── exp_00.py
│       └── ...
├── outputs/              # Experimental results
└── tests/                # E8e verification tests
```

---

### Citation

If you use this code, please cite the relevant E8 papers:

**E8a (Observation):**
```bibtex
@preprint{hideki2025e8a,
  title={Forgetting Mechanics E8a: Method of Loci - Spatialized Blueprint},
  author={HIDEKI},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17541046}
}
```

**E8b (Laws):**
```bibtex
@preprint{hideki2025e8b,
  title={Forgetting Mechanics E8b: Birth of Structural Optics - Laws of Orthogonality and Coupling},
  author={HIDEKI},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17549824}
}
```

**E8e (Verification):**
```bibtex
@software{hideki2025e8e,
  title={Forgetting Mechanics E8e: Code Validity Verification},
  author={HIDEKI},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17569347}
}
```

**GitHub Repository:**
```bibtex
@software{hideki2025optics,
  title={Optics of Intelligence: E8 Experiments Code Repository},
  author={HIDEKI},
  year={2025},
  publisher={GitHub},
  url={https://github.com/HIDEKI-SQ/optics-of-intelligence}
}
```

For complete citation list, see: https://zenodo.org/communities/kakushin-structural-theory/

---

### License

**Code:** MIT License  
**Documentation:** CC BY 4.0

See [`LICENSE`](LICENSE) for details.

---

### Contact

**HIDEKI**  
Independent Researcher  
ORCID: [0009-0002-0019-6608](https://orcid.org/0009-0002-0019-6608)  
Email: hideki@r3776.jp

---

<a name="japanese"></a>
## 日本語

### 論文

- **E8a** (2025-137): 観測 - VS≈0の発見（自然直交性）
- **E8b** (2025-138): 法則 - O1-O4普遍法則の確立
- **E8c** (2025-139): 方程式 - 幾何学的定式化（V=ν⊙ψ, M=Π_V(S)）
- **E8d** (2025-140): 統合 - 知性の相対性理論
- **E8e** (2025-141): 検証 - コード妥当性と再現性
- **E8f** (2025-142): 完成 - 完全決定論的実装

全プレプリントは [Zenodo 核信構造論コミュニティ](https://zenodo.org/communities/kakushin-structural-theory/) で公開されている。

### クイックスタート

#### ⚡ 方法1: 全実験の実行（再現研究に推奨）

E8研究の完全な再現のため:
```bash
# リポジトリをクローン
git clone https://github.com/HIDEKI-SQ/optics-of-intelligence.git
cd optics-of-intelligence

# 依存関係をインストール
pip install -r requirements.txt

# 全実験を実行（β, exp00-13）
python run_all.py
```

結果は `outputs/` に保存され、ハッシュ検証は `outputs/hash_manifest.json` に記録される。

**所要時間:** ハードウェアに依存して10-30分。

---

#### 🎓 方法2: 単一実験（Google Colab）

迅速な探索または教育目的のため:

1. [`colab/`](colab/) フォルダに移動
2. 任意の `exp0X_standalone.py` ファイルを開く
3. 全内容をコピー
4. Google Colabのセルに貼り付け
5. 「実行」をクリック（Ctrl+Enter または Cmd+Enter）

**例:** [`colab/exp01_standalone.py`](colab/exp01_standalone.py) で、記憶の宮殿の2000年来の仮定に挑戦したVS≈0現象を観測できる。

**インストール不要** - 全依存関係はGoogle Colabに標準搭載されている。

---

#### 🔬 方法3: 高度な使用法

実験の拡張または修正のため:
```python
from src.experiments.exp_01 import run_exp01

# カスタムシードで実行
results = run_exp01(seed=123)

# 特定の指標にアクセス
print(f"VS (spatial): {results['vs_spatial']:.3f}")
print(f"VS (random): {results['vs_random']:.3f}")
```

モジュール構造については [`src/`](src/) ディレクトリを参照。

---

### 主要な発見

**自然直交性（O1）:**
価値圧力（λ=0）が存在しない場合、構造と意味は独立しており、全条件でVS≈0となる。

**位相優位性（O2）:**
構造保存は、計量幾何ではなく位相（φ）に依存する。

**ストレス耐性（O3）:**
構造的破壊は、自動的に意味的混乱を引き起こさない。

**価値ゲート結合（O4）:**
価値（λ>0）のみが構造と意味を結合し、VS>0を実現する。

---

### システム要件

- Python 3.11以上
- numpy, scipy, pandas, matplotlib, scikit-learn
- 正確なバージョンは [`requirements.txt`](requirements.txt) を参照

---

### リポジトリ構造
```
optics-of-intelligence/
├── README.md              # 本ファイル
├── LICENSE                # MITライセンス
├── requirements.txt       # Python依存関係
├── run_all.py            # 全実験の実行
├── colab/                # Google Colab用の単独ファイル
│   ├── exp01_standalone.py
│   └── ...
├── src/                  # モジュール化されたソースコード
│   ├── core/             # コア機能
│   │   ├── config.py     # 設定
│   │   ├── measures.py   # SP/GEN/VS指標
│   │   ├── utils.py      # ユーティリティ
│   │   └── hash_recorder.py
│   └── experiments/      # 個別実験
│       ├── exp_beta.py
│       ├── exp_00.py
│       └── ...
├── outputs/              # 実験結果
└── tests/                # E8e検証テスト
```

---

### 引用

このコードを使用する場合は、関連するE8論文を引用されたい：

**E8a（観測）:**
```bibtex
@preprint{hideki2025e8a,
  title={忘却の構造力学 E8a: 記憶の宮殿 - 空間化されたBlueprint},
  author={HIDEKI},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17541046}
}
```

**E8b（法則）:**
```bibtex
@preprint{hideki2025e8b,
  title={忘却の構造力学 E8b: 構造光学の誕生 - 直交性と結合の法則},
  author={HIDEKI},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17549824}
}
```

**E8e（検証）:**
```bibtex
@software{hideki2025e8e,
  title={忘却の構造力学 E8e: コード妥当性検証},
  author={HIDEKI},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17569347}
}
```

**GitHubリポジトリ:**
```bibtex
@software{hideki2025optics,
  title={知性光学：E8実験コードリポジトリ},
  author={HIDEKI},
  year={2025},
  publisher={GitHub},
  url={https://github.com/HIDEKI-SQ/optics-of-intelligence}
}
```

完全な引用リストは以下を参照： https://zenodo.org/communities/kakushin-structural-theory/

---

### ライセンス

**コード:** MITライセンス  
**ドキュメント:** CC BY 4.0

詳細は [`LICENSE`](LICENSE) を参照。

---

### 連絡先

**HIDEKI**  
独立研究者  
ORCID: [0009-0002-0019-6608](https://orcid.org/0009-0002-0019-6608)  
Email: hideki@r3776.jp
