"""
Scraper de resenhas de livros do Skoob.
Coleta resenhas textuais para posterior análise de NLP.
"""

import json
import re
import logging
import csv
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from scraper.skoob_client import SkoobClient

logger = logging.getLogger("ReviewScraper")


class ReviewScraper:
    """
    Scraper de resenhas de livros do Skoob.

    Coleta resenhas textuais, notas e datas para análise de NLP.
    URL: https://www.skoob.com.br/pt/book/{book_id}/reviews
    """

    REVIEWS_URL = "https://www.skoob.com.br/pt/book/{book_id}/reviews"

    def __init__(self):
        self.client = SkoobClient()
        self.scraped_reviews = []

    def scrape_reviews(
        self, book_id: int, max_reviews: int = None
    ) -> list:
        """
        Coleta resenhas de um livro específico.

        Args:
            book_id: ID do livro
            max_reviews: Limite de resenhas (usa config se None)

        Returns:
            Lista de dicionários com os dados das resenhas
        """
        if max_reviews is None:
            max_reviews = config.MAX_REVIEWS_PER_BOOK

        url = self.REVIEWS_URL.format(book_id=book_id)
        reviews = []
        page = 1

        while len(reviews) < max_reviews:
            html = self.client.get_html(url, params={"page": page})
            if html is None:
                break

            # Verificar redirecionamento para login
            if "/login" in html[:500]:
                logger.warning(f"Livro {book_id}: resenhas requerem login")
                break

            page_reviews = self._extract_reviews(html, book_id)

            if not page_reviews:
                break

            reviews.extend(page_reviews)
            page += 1

            if len(page_reviews) < 10:  # Menos resultados que o esperado = última página
                break

        # Limitar ao máximo configurado
        reviews = reviews[:max_reviews]
        logger.info(f"Livro {book_id}: {len(reviews)} resenhas coletadas")

        return reviews

    def _extract_reviews(self, html: str, book_id: int) -> list:
        """Extrai resenhas do HTML de uma página."""
        soup = BeautifulSoup(html, "html.parser")
        reviews = []

        # Tentar extrair do __NEXT_DATA__ primeiro (mais estruturado)
        nd = soup.find("script", id="__NEXT_DATA__")
        if nd and nd.string:
            try:
                data = json.loads(nd.string)
                pp = data.get("props", {}).get("pageProps", {})
                review_list = pp.get("reviews", pp.get("data", []))

                if isinstance(review_list, list):
                    for item in review_list:
                        if isinstance(item, dict):
                            review = {
                                "book_id": book_id,
                                "usuario": item.get("user", {}).get("name", "")
                                if isinstance(item.get("user"), dict)
                                else str(item.get("user", "")),
                                "texto": item.get("text", item.get("content", "")),
                                "nota": item.get("rating", item.get("score", "")),
                                "data": item.get("date", item.get("created_at", "")),
                                "curtidas": item.get("likes", item.get("likes_count", 0)),
                            }
                            if review["texto"]:  # Só adicionar se tem texto
                                reviews.append(review)

                if reviews:
                    return reviews
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: parsing de HTML
        # Procurar blocos de resenha na página
        review_blocks = soup.find_all("div", class_=re.compile(r"review|resenha", re.I))
        if not review_blocks:
            # Tentar encontrar por estrutura
            review_blocks = soup.find_all("article")

        for block in review_blocks:
            # Texto da resenha
            text_el = block.find("p") or block.find("div", class_=re.compile(r"text|content", re.I))
            if not text_el:
                continue

            texto = text_el.get_text(strip=True)
            if len(texto) < 20:  # Ignorar textos muito curtos
                continue

            # Nota/estrelas
            nota = ""
            star_el = block.find(class_=re.compile(r"star|rating|nota", re.I))
            if star_el:
                nota_text = star_el.get_text(strip=True)
                match = re.search(r"(\d+)", nota_text)
                if match:
                    nota = int(match.group(1))

            # Data
            data_str = ""
            date_el = block.find("time") or block.find(class_=re.compile(r"date|data", re.I))
            if date_el:
                data_str = date_el.get("datetime", date_el.get_text(strip=True))

            # Usuário
            user_el = block.find("a", href=re.compile(r"/user/|/perfil/|/reader/"))
            usuario = user_el.get_text(strip=True) if user_el else ""

            review = {
                "book_id": book_id,
                "usuario": usuario,
                "texto": texto,
                "nota": nota,
                "data": data_str,
                "curtidas": 0,
            }
            reviews.append(review)

        return reviews

    def scrape_reviews_for_books(
        self, book_ids: list, output_file: str = None, max_reviews_per_book: int = None
    ):
        """
        Coleta resenhas para uma lista de livros.

        Args:
            book_ids: Lista de IDs de livros
            output_file: Caminho do CSV de saída
            max_reviews_per_book: Limite de resenhas por livro
        """
        if output_file is None:
            output_file = config.RAW_REVIEWS_FILE

        all_reviews = []

        for book_id in tqdm(book_ids, desc="Coletando resenhas"):
            reviews = self.scrape_reviews(book_id, max_reviews_per_book)
            all_reviews.extend(reviews)
            self.scraped_reviews.extend(reviews)

            # Salvar periodicamente
            if len(all_reviews) % 200 == 0 and all_reviews:
                self._save_to_csv(all_reviews, output_file)

        # Salvar resultado final
        self._save_to_csv(all_reviews, output_file)
        logger.info(
            f"Coleta de resenhas finalizada! "
            f"{len(all_reviews)} resenhas de {len(book_ids)} livros "
            f"salvos em {output_file}"
        )

        return all_reviews

    def _save_to_csv(self, reviews: list, output_file: str):
        """Salva resenhas em CSV."""
        if not reviews:
            return

        fieldnames = ["book_id", "usuario", "texto", "nota", "data", "curtidas"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reviews)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  ReviewScraper - Teste de Coleta")
    print("=" * 60)

    with ReviewScraper() as scraper:
        # Teste com livro ID=1 (Ensaio sobre a Cegueira)
        reviews = scraper.scrape_reviews(1, max_reviews=5)
        print(f"\nResenhas encontradas: {len(reviews)}")
        for r in reviews[:3]:
            print(f"\n  Usuário: {r['usuario']}")
            print(f"  Nota: {r['nota']}")
            print(f"  Texto: {r['texto'][:150]}...")

    print("\nTeste concluído!")
