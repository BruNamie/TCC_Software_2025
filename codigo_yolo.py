import os, boto3, cv2
from ultralytics import YOLO
from botocore.exceptions import ClientError
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal

BUCKET = os.environ.get("BUCKET_NAME", "fotos-esp32")
INPUT_PREFIXO = os.environ.get("INPUT_IMAGE_PREFIXO", "")   # processa todo o bucket se não for definido
OUTPUT_PREFIXO = os.environ.get("OUTPUT_IMAGE_PREFIXO", "resultados/")
NOME_MODELO = os.environ.get("NOME_MODELO", "yolov8n.pt")
NOME_TABELA = os.environ.get("DYNAMO_TABLE", "yolo-resultados")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "sa-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)
dynamo_db = boto3.resource("dynamodb", region_name=AWS_REGION)
tabela = dynamo_db.Table(NOME_TABELA)


def lista_imagens(bucket, prefix=""):
    """
    Lista todas as imagens em um bucket/prefixo.
    """
    lista_img = []
    paginator = s3.get_paginator("list_objects_v2")
    for pag in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in pag.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".jpg", ".png")):
                lista_img.append(key)
    return lista_img


def download_s3(bucket, key, local_path):
    """
    Faz o download de um arquivo do bucket S3 para o caminho local.

    Parâmetros:
        bucket (str): Nome do bucket S3.
        key (str): Caminho/Key do arquivo no S3.
        local_path (str): Caminho local onde o arquivo será salvo.
    """
    s3.download_file(bucket, key, local_path)


def upload_s3(local_path, bucket, key):
    """
    Faz o upload de um arquivo local para o bucket S3 especificado.

    Parâmetros:
        local_path (str): Caminho do arquivo local a ser enviado.
        bucket (str): Nome do bucket S3.
        key (str): Caminho/Key onde o arquivo será salvo no S3.
    """
    s3.upload_file(local_path, bucket, key)


def pega_metadados(bucket, key):
    """
    Retorna metadados da imagem.

    Parâmetros:
        bucket (str): Nome do bucket S3.
        key (str): Caminho/Key do objeto no S3.
    """
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        metadados = head.get("Metadata", {})
        return metadados
    except ClientError as e:
        print(f"Falha ao obter metadados de {key}: {e}")
        return {}


def salva_dynamo(imagem_key, output_key, n_pessoas, metadados, hora_execucao):
    """
    Salva os resultados na tabela do DynamoDB.

    Parâmetros:
        imagem_key (str): Caminho da imagem original no S3.
        output_key (str): Caminho da imagem processada no S3.
        n_pessoas (int): Número de pessoas detectadas.
        metadados (dict): Metadados associados à imagem.
        hora_execucao (str): Timestamp da execução.
    """
    item = {
        "imagem": imagem_key,
        "outputKey": output_key,
        "qtd_pessoas": int(n_pessoas),
        "bateria": None,
        "hora_imagem": None,
        "hora_execucao": hora_execucao
    }
    if metadados.get("battery-level"):
        item["bateria"] = Decimal(str(metadados["battery-level"]))
    if metadados.get("device-timestamp"):
        item["hora_imagem"] = metadados["device-timestamp"]

    try:
        tabela.put_item(Item=item)
        print(f"Gravado no DynamoDB: {imagem_key}")
    except ClientError as e:
        print(f"Falha ao gravar no DynamoDB: {e}")
        raise


def processa_imagem(input_key):
    """
    Executa o processamento da imagem:
    - Faz download da imagem do S3
    - Detecta pessoas na imagem
    - Faz upload da imagem com identificação de pessoas para o S3
    - Salva os resultados no DynamoDB

    Parâmetros:
        input_key (str): Caminho/Key da imagem de entrada no S3.
    """
    # Caminhos locais temporários
    local_input = os.path.basename(input_key)
    local_output = "out_" + local_input

    print(f"Baixando s3://{BUCKET}/{input_key}")
    download_s3(BUCKET, input_key, local_input)

    # Carrega e processa imagem
    model = YOLO(NOME_MODELO)
    img = cv2.imread(local_input)
    img = cv2.rotate(img, cv2.ROTATE_180)
    results = model.predict(source=img, conf=0.15, classes=[0]) 
    qtd_pessoas = len(results[0].boxes)
    
    # Salva imagem com identificação de pessoas
    img_final = results[0].plot()
    cv2.imwrite(local_output, img_final)

    # Upload da imagem anotada
    output_key = OUTPUT_PREFIXO + os.path.basename(local_output)
    print(f"Enviando {local_output} → s3://{BUCKET}/{output_key}")
    upload_s3(local_output, BUCKET, output_key)

    # Metadados + hora
    metadados = pega_metadados(BUCKET, input_key)
    hora_execucao = datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()

    # DynamoDB
    salva_dynamo(
        imagem_key=input_key,
        output_key=output_key,
        n_pessoas=qtd_pessoas,
        metadados=metadados,
        hora_execucao=hora_execucao
    )

    # Limpeza de arquivos locais
    os.remove(local_input)
    os.remove(local_output)


def main():
    print(f"Listando imagens em s3://{BUCKET}/{INPUT_PREFIXO}")
    imagens = lista_imagens(BUCKET, INPUT_PREFIXO)

    if not imagens:
        print("Nenhuma imagem encontrada no bucket.")
        return

    for img in imagens:
        try:
            processa_imagem(img)
        except Exception as e:
            print(f"Erro ao processar {img}: {e}")


if __name__ == "__main__":
    try:
        main()
    except ClientError as e:
        print(f"Erro S3: {e}")
        raise
    except Exception as e:
        print(f"Erro: {e}")
        raise
