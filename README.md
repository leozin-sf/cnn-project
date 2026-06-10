# Classificacao Multi-Label de Radiografias com CNN

Projeto de aprendizado profundo para classificar 14 condicoes em radiografias
de torax do dataset **NIH Chest X-ray 14**. O notebook utiliza transfer learning
com a arquitetura DenseNet121 e TensorFlow/Keras.

## Estrutura

```text
.
├── .gitignore
├── projeto3_cnn_chestxray.ipynb
├── requirements-cpu.txt
├── requirements-rocm.txt
├── requirements.txt
├── scripts/
│   └── setup_rocm_wsl.sh
└── README.md
```

## Pre-requisitos

- Python 3.12 para usar a RX 9070 XT com o wheel oficial do TensorFlow ROCm
- Python 3.10, 3.11 ou 3.12 para executar somente em CPU
- VS Code
- Extensoes do VS Code:
  - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- Conta no [Kaggle](https://www.kaggle.com/)
- Aproximadamente 100 GB livres para manter o arquivo compactado e o dataset
  extraido durante a preparacao

O treinamento completo e pesado. Este projeto inclui configuracao para a
**AMD Radeon RX 9070 XT** usando ROCm no WSL2.

## Criar e ativar a venv

Abra um terminal na raiz do projeto.

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativacao, execute uma vez na sessao atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

Quando a venv estiver ativa, o terminal exibira `(.venv)` no inicio da linha.
Para desativa-la:

```bash
deactivate
```

## Instalar para CPU

Com a venv ativa:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

## Configurar a RX 9070 XT

O TensorFlow padrao procura CUDA, que funciona apenas com NVIDIA. Para usar a
RX 9070 XT, o WSL2 precisa expor `/dev/dxg` e o ambiente deve usar TensorFlow
compilado para ROCm.

### 1. Reiniciar o WSL

Feche o VS Code e todas as janelas do Ubuntu. No PowerShell do Windows:

```powershell
wsl --shutdown
```

Abra novamente o Ubuntu e valide:

```bash
ls -l /dev/dxg
```

Se `/dev/dxg` continuar ausente, instale o
[AMD Adrenalin 26.2.2 com suporte a WSL2](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html),
reinicie o Windows e repita esta etapa.

### 2. Instalar ROCm, ROCDXG e TensorFlow

Na raiz do projeto:

```bash
bash scripts/setup_rocm_wsl.sh
```

O script solicitará sua senha do Ubuntu para instalar pacotes do sistema. Ele:

- instala ROCm 7.2.x;
- compila a integração ROCDXG;
- substitui o TensorFlow padrão pelo TensorFlow ROCm 2.20;
- valida se a RX 9070 XT aparece para o TensorFlow.

### 3. Validar manualmente

Depois da instalação:

```bash
source .venv/bin/activate
HSA_ENABLE_DXG_DETECTION=1 python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

O resultado esperado deve conter pelo menos um `PhysicalDevice` do tipo `GPU`.
Não execute `pip install tensorflow`, pois isso substituirá a versão ROCm.

### Melhorar o uso da GPU

O pipeline usa `tf.data`, processamento paralelo, batch 64, prefetch,
augmentation na GPU e precisão mista (`mixed_float16`).

O arquivo `%USERPROFILE%\.wslconfig` pode limitar os processadores disponíveis
para o carregamento das imagens. Para o Ryzen 7 5700X3D, use:

```ini
[wsl2]
memory=16GB
processors=16
swap=6GB
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual
```

Depois de alterar o arquivo, aplique a configuração no PowerShell:

```powershell
wsl --shutdown
```

O uso da CPU continuará significativo porque leitura e decodificação de PNG
são realizadas nela. No Gerenciador de Tarefas do Windows, selecione um gráfico
de **Compute** da GPU; o gráfico padrão de **3D** não representa corretamente o
uso do TensorFlow.

## Configurar a API do Kaggle

1. Acesse as configuracoes da sua conta no Kaggle.
2. Gere o token da API e baixe o arquivo `kaggle.json`.
3. Coloque o arquivo no diretorio esperado pelo Kaggle.

No Linux, macOS ou WSL:

```bash
mkdir -p ~/.kaggle
cp /caminho/para/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets files nih-chest-xrays/data
```

No Windows, coloque o arquivo em:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

Nao adicione `kaggle.json` ao Git, pois ele contem suas credenciais.

## Executar no VS Code

1. Abra a pasta do projeto no VS Code:

   ```bash
   code .
   ```

2. Abra o arquivo `projeto3_cnn_chestxray.ipynb`.
3. Clique em **Select Kernel** no canto superior direito.
4. Selecione **Python Environments** e depois o interpretador `.venv`.
5. Execute as celulas em ordem.

Caso a `.venv` nao apareca na lista de kernels:

```bash
python -m ipykernel install --user --name cnn-project --display-name "Python (.venv - cnn-project)"
```

Depois selecione o kernel **Python (.venv - cnn-project)**.

## Execucao local

O notebook ja esta adaptado para o VS Code:

- nao importa `google.colab`;
- procura a credencial em `~/.kaggle/kaggle.json`;
- baixa o dataset em `data/nih_chestxray`;
- extrai os arquivos usando o modulo `zipfile` do Python;
- procura as imagens recursivamente dentro de `data/nih_chestxray`;
- salva modelos e figuras no diretorio `outputs`.

Antes de executar a celula de download, confirme a autenticacao:

```bash
source .venv/bin/activate
kaggle datasets files nih-chest-xrays/data
```

O download completo tem aproximadamente 42 GB. A celula evita baixar novamente
arquivos que ja existem e so remove o ZIP principal depois de confirmar que as
imagens foram extraidas.

## Observacoes

- O download completo do NIH Chest X-ray 14 tem dezenas de gigabytes e pode
  levar bastante tempo.
- A primeira criacao da DenseNet121 tambem baixa os pesos do ImageNet.
- Os arquivos de modelo, imagens e dataset nao devem ser versionados no Git.
- Este projeto e academico e nao substitui avaliacao ou diagnostico medico.
