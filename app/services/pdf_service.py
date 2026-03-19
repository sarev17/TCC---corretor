"""
Serviço responsável por converter PDFs em imagens.

Utiliza pdf2image para converter cada página em imagem PIL.
"""

from pdf2image import convert_from_path
from typing import List
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Converte um PDF em lista de imagens (uma por página).

    :param pdf_path: caminho do arquivo PDF
    :type pdf_path: str

    :param dpi: resolução da conversão
    :type dpi: int

    :return: lista de imagens PIL
    :rtype: List[PIL.Image.Image]
    """

    try:
        images = convert_from_path(pdf_path, dpi=dpi)

        if not images:
            raise Exception("Nenhuma página encontrada no PDF")

        return images

    except Exception as e:
        raise Exception(f"Erro ao converter PDF: {str(e)}")