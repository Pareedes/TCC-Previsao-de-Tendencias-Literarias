"""
Scraper de rankings e busca de livros por gênero no Skoob.
Coleta IDs de livros a partir de páginas de busca e categorias.
"""

import json
import re
import logging
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from scraper.skoob_client import SkoobClient

logger = logging.getLogger("RankingScraper")


class RankingScraper:
    """
    Scraper para coletar IDs de livros a partir de rankings e buscas no Skoob.

    Estratégia: buscar livros por gênero e por popularidade para construir
    uma lista diversificada de IDs que depois serão detalhados pelo BookScraper.
    """

    SEARCH_URL = "https://www.skoob.com.br/pt/livros"
    EXPLORE_URL = "https://www.skoob.com.br/pt/explore"
    GENRE_URL = "https://www.skoob.com.br/pt/tag/{genre_slug}"

    # Mapeamento de gêneros para slugs de URL
    GENRE_SLUGS = {
        "Romance": "romance",
        "Fantasia": "fantasia",
        "Ficção Científica": "ficcao-cientifica",
        "Terror": "terror",
        "Suspense": "suspense",
        "Mistério": "misterio",
        "Drama": "drama",
        "Aventura": "aventura",
        "Infantojuvenil": "infantojuvenil",
        "Jovem Adulto": "jovem-adulto",
        "Poesia": "poesia",
        "Biografia": "biografia",
        "Autoajuda": "autoajuda",
        "História": "historia",
        "Humor": "humor",
        "Clássicos": "classicos",
        "Distopia": "distopia",
        "HQ": "hq",
        "Mangá": "manga",
        "Religião": "religiao",
        "Policial": "policial",
        "Crônica": "cronica",
        "Conto": "conto",
        "Filosofia": "filosofia",
        "Psicologia": "psicologia",
        "Negócios": "negocios",
    }

    def __init__(self):
        self.client = SkoobClient()
        self.collected_ids = set()

    def collect_ids_by_genre(self, genre: str, max_pages: int = 10) -> list:
        """
        Coleta IDs de livros de um gênero específico.

        Args:
            genre: Nome do gênero (ex: "Romance")
            max_pages: Número máximo de páginas a percorrer

        Returns:
            Lista de IDs de livros encontrados
        """
        slug = self.GENRE_SLUGS.get(genre, genre.lower().replace(" ", "-"))
        genre_ids = []

        for page in range(1, max_pages + 1):
            url = self.GENRE_URL.format(genre_slug=slug)
            params = {"page": page}

            html = self.client.get_html(url, params)
            if html is None:
                break

            ids = self._extract_book_ids(html)
            if not ids:
                logger.info(f"Gênero '{genre}' - Página {page}: sem mais resultados")
                break

            new_ids = [bid for bid in ids if bid not in self.collected_ids]
            genre_ids.extend(new_ids)
            self.collected_ids.update(new_ids)

            logger.info(
                f"Gênero '{genre}' - Página {page}: {len(new_ids)} novos IDs "
                f"(total gênero: {len(genre_ids)})"
            )

        return genre_ids

    def collect_ids_from_explore(self, max_pages: int = 20) -> list:
        """
        Coleta IDs de livros da página de exploração (mais populares).

        Returns:
            Lista de IDs de livros encontrados
        """
        explore_ids = []

        for page in range(1, max_pages + 1):
            html = self.client.get_html(self.EXPLORE_URL, params={"page": page})
            if html is None:
                break

            ids = self._extract_book_ids(html)
            if not ids:
                break

            new_ids = [bid for bid in ids if bid not in self.collected_ids]
            explore_ids.extend(new_ids)
            self.collected_ids.update(new_ids)

            logger.info(
                f"Explorar - Página {page}: {len(new_ids)} novos IDs "
                f"(total: {len(explore_ids)})"
            )

        return explore_ids

    def collect_ids_sequential(self, start: int = 1, end: int = 15000) -> list:
        """
        Gera IDs sequenciais para coleta direta.
        Abordagem mais simples: testar IDs sequencialmente.

        Args:
            start: ID inicial
            end: ID final

        Returns:
            Lista de IDs
        """
        logger.info(f"Gerando IDs sequenciais de {start} a {end}")
        return list(range(start, end + 1))

    def collect_all_genre_ids(
        self, genres: list = None, max_pages_per_genre: int = 5
    ) -> dict:
        """
        Coleta IDs de livros de todos os gêneros configurados.

        Args:
            genres: Lista de gêneros (usa GENEROS_ALVO se None)
            max_pages_per_genre: Páginas máximas por gênero

        Returns:
            Dicionário {gênero: [lista de IDs]}
        """
        if genres is None:
            genres = config.GENEROS_ALVO

        genre_ids = {}

        for genre in tqdm(genres, desc="Coletando IDs por gênero"):
            ids = self.collect_ids_by_genre(genre, max_pages_per_genre)
            genre_ids[genre] = ids
            logger.info(f"Gênero '{genre}': {len(ids)} IDs coletados")

        total = sum(len(ids) for ids in genre_ids.values())
        unique = len(self.collected_ids)
        logger.info(
            f"Coleta por gênero finalizada: {total} IDs totais, {unique} únicos"
        )

        return genre_ids

    def _extract_book_ids(self, html: str) -> list:
        """
        Extrai IDs de livros do HTML de uma página de listagem.

        Procura links no formato /pt/book/{id} ou /book/{id}
        """
        soup = BeautifulSoup(html, "html.parser")
        book_ids = []

        # Procurar links para páginas de livros
        book_links = soup.find_all("a", href=re.compile(r"/book/(\d+)"))
        for link in book_links:
            href = link.get("href", "")
            match = re.search(r"/book/(\d+)", href)
            if match:
                book_id = int(match.group(1))
                if book_id not in book_ids:
                    book_ids.append(book_id)

        # Também procurar no formato antigo /livro/SLUG-ID
        old_links = soup.find_all("a", href=re.compile(r"-(\d+)\.html"))
        for link in old_links:
            href = link.get("href", "")
            match = re.search(r"-(\d+)\.html", href)
            if match:
                book_id = int(match.group(1))
                if book_id not in book_ids:
                    book_ids.append(book_id)

        # Procurar em __NEXT_DATA__
        nd = soup.find("script", id="__NEXT_DATA__")
        if nd and nd.string:
            try:
                data = json.loads(nd.string)
                page_props = data.get("props", {}).get("pageProps", {})

                # Tentar encontrar listas de livros em diferentes formatos
                for key in ["books", "items", "results", "data"]:
                    items = page_props.get(key, [])
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                bid = item.get("id", item.get("book_id", None))
                                if bid and isinstance(bid, int) and bid not in book_ids:
                                    book_ids.append(bid)
            except (json.JSONDecodeError, KeyError):
                pass

        return book_ids

    def save_ids(self, ids: list | dict, output_file: str = None):
        """
        Salva IDs coletados em arquivo JSON.

        Args:
            ids: Lista de IDs ou dicionário {gênero: [IDs]}
            output_file: Caminho do arquivo
        """
        if output_file is None:
            output_file = os.path.join(config.RAW_DATA_DIR, "book_ids.json")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)

        logger.info(f"IDs salvos em: {output_file}")

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  RankingScraper - Teste de Coleta de IDs")
    print("=" * 60)

    with RankingScraper() as scraper:
        # Teste: buscar IDs de 2 gêneros
        for genre in ["Romance", "Fantasia"]:
            print(f"\nBuscando IDs do gênero '{genre}'...")
            ids = scraper.collect_ids_by_genre(genre, max_pages=1)
            print(f"  IDs encontrados: {len(ids)}")
            if ids:
                print(f"  Primeiros IDs: {ids[:5]}")

    print("\nTeste concluído!")
