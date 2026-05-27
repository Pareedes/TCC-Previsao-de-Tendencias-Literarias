"""
Scraper de dados de livros do Skoob.
Coleta informações de livros individuais via parsing de HTML e JSON-LD.
Suporta scraping concorrente com ThreadPoolExecutor para maior velocidade.

URL base funcional (sem necessidade de login): https://www.skoob.com.br/pt/book/{id}
"""

import json
import re
import logging
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from scraper.skoob_client import SkoobClient

logger = logging.getLogger("BookScraper")

# Lock para escrita thread-safe em listas e arquivos
_lock = threading.Lock()


class BookScraper:
    """
    Scraper de dados de livros individuais do Skoob.

    Extrai dados de cada livro a partir da página pública /pt/book/{id}:
    - Dados estruturados via JSON-LD (Schema.org Book)
    - Dados adicionais via parsing HTML (leitores, resenhas, gêneros, etc.)
    """

    BOOK_URL_TEMPLATE = "https://www.skoob.com.br/pt/book/{book_id}"

    def __init__(self):
        self.client = SkoobClient()
        self.scraped_books = []

    def scrape_book(self, book_id: int) -> dict | None:
        """
        Coleta todos os dados de um livro pelo seu ID.

        Args:
            book_id: ID do livro no Skoob

        Returns:
            Dicionário com os dados do livro, ou None se não encontrado
        """
        url = self.BOOK_URL_TEMPLATE.format(book_id=book_id)
        html = self.client.get_html(url)

        if html is None:
            return None

        # Verificar se foi redirecionado para login
        if "/login" in html[:500]:
            logger.warning(f"Livro {book_id}: redirecionado para login, pulando")
            return None

        soup = BeautifulSoup(html, "html.parser")

        # Verificar se a página é válida (livro existe)
        title_tag = soup.find("title")
        if title_tag and title_tag.string and title_tag.string.strip() == "Skoob":
            logger.info(f"Livro {book_id}: página genérica (livro não encontrado)")
            return None

        book_data = {"book_id": book_id, "url": url}

        # 1) Extrair dados do JSON-LD (fonte mais confiável)
        self._extract_json_ld(soup, book_data)

        # 2) Extrair dados do HTML
        self._extract_html_data(soup, book_data)

        # 3) Extrair dados do __NEXT_DATA__ se disponível
        self._extract_next_data(soup, book_data)

        if not book_data.get("titulo"):
            logger.warning(f"Livro {book_id}: não foi possível extrair título")
            return None

        return book_data

    def _extract_json_ld(self, soup: BeautifulSoup, book_data: dict):
        """Extrai dados do script JSON-LD (Schema.org)."""
        ld_script = soup.find("script", type="application/ld+json")
        if not ld_script or not ld_script.string:
            return

        try:
            ld_data = json.loads(ld_script.string)
            if ld_data.get("@type") == "Book":
                book_data["titulo"] = ld_data.get("name", "")
                
                # Autor
                author = ld_data.get("author", {})
                if isinstance(author, dict):
                    book_data["autor"] = author.get("name", "")
                elif isinstance(author, list) and len(author) > 0:
                    book_data["autor"] = author[0].get("name", "")

                # Imagem da capa
                book_data["imagem_capa"] = ld_data.get("image", "")

                # Descrição/sinopse
                book_data["sinopse"] = ld_data.get("description", "")

                # Avaliação (rating)
                rating = ld_data.get("aggregateRating", {})
                if rating:
                    book_data["nota_media"] = float(rating.get("ratingValue", 0))
                    book_data["total_avaliacoes"] = int(rating.get("ratingCount", 0))

                # ISBN
                book_data["isbn"] = ld_data.get("isbn", "")

                # Editora
                publisher = ld_data.get("publisher", {})
                if isinstance(publisher, dict):
                    book_data["editora"] = publisher.get("name", "")

                # Número de páginas
                book_data["num_paginas"] = ld_data.get("numberOfPages", "")

                # Data de publicação
                book_data["data_publicacao"] = ld_data.get("datePublished", "")

                # Gênero do JSON-LD
                genre = ld_data.get("genre", "")
                if genre:
                    if isinstance(genre, list):
                        book_data["generos"] = ", ".join(genre)
                    else:
                        book_data["generos"] = genre

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Erro ao parsear JSON-LD: {e}")

    def _extract_html_data(self, soup: BeautifulSoup, book_data: dict):
        """Extrai dados adicionais do HTML da página."""

        # Título (fallback - do H1)
        if not book_data.get("titulo"):
            h1 = soup.find("h1")
            if h1:
                book_data["titulo"] = h1.get_text(strip=True)

        # Estatísticas de leitores (via links com filtros)
        filter_mapping = {
            "filter=read": "leitores_leram",
            "filter=reading": "leitores_lendo",
            "filter=want_to_read": "leitores_quero_ler",
            "filter=rereading": "leitores_relendo",
            "filter=abandoned": "leitores_abandonaram",
        }

        for filter_key, field_name in filter_mapping.items():
            link = soup.find("a", href=re.compile(re.escape(filter_key)))
            if link:
                # O número geralmente está em um span dentro do link
                spans = link.find_all("span")
                for span in spans:
                    text = span.get_text(strip=True).replace(".", "").replace(",", "")
                    if text.isdigit():
                        book_data[field_name] = int(text)
                        break

        # Total de leitores (soma)
        total_leitores = 0
        for field in filter_mapping.values():
            total_leitores += book_data.get(field, 0)
        if total_leitores > 0:
            book_data["total_leitores"] = total_leitores

        # Número de resenhas (via link de resenhas)
        review_link = soup.find("a", href=re.compile(r"/reviews"))
        if review_link:
            spans = review_link.find_all("span")
            for span in spans:
                text = span.get_text(strip=True).replace(".", "").replace(",", "")
                if text.isdigit():
                    book_data["total_resenhas"] = int(text)
                    break

        # Número de edições
        editions_link = soup.find("a", href=re.compile(r"/editions"))
        if editions_link:
            spans = editions_link.find_all("span")
            for span in spans:
                text = span.get_text(strip=True).replace(".", "").replace(",", "")
                if text.isdigit():
                    book_data["total_edicoes"] = int(text)
                    break

        # Gêneros/Tags (podem estar em elementos com texto de gênero)
        if not book_data.get("generos"):
            # Procurar tags ou categorias na página
            tag_elements = soup.find_all("a", href=re.compile(r"/tag/|/genre/|/category/"))
            if tag_elements:
                tags = [t.get_text(strip=True) for t in tag_elements if t.get_text(strip=True)]
                if tags:
                    book_data["generos"] = ", ".join(tags)

    def _extract_next_data(self, soup: BeautifulSoup, book_data: dict):
        """Extrai dados do __NEXT_DATA__ (dados do Next.js SSR)."""
        nd = soup.find("script", id="__NEXT_DATA__")
        if not nd or not nd.string:
            return

        try:
            data = json.loads(nd.string)
            page_props = data.get("props", {}).get("pageProps", {})

            # Tentar extrair dados do book
            book = page_props.get("book", page_props.get("data", {}))
            if isinstance(book, dict):
                # Preencher campos que ainda não foram extraídos
                if not book_data.get("titulo"):
                    book_data["titulo"] = book.get("title", book.get("name", ""))
                if not book_data.get("autor"):
                    author = book.get("author", {})
                    if isinstance(author, dict):
                        book_data["autor"] = author.get("name", "")
                    elif isinstance(author, str):
                        book_data["autor"] = author
                if not book_data.get("generos"):
                    genres = book.get("genres", book.get("tags", []))
                    if isinstance(genres, list):
                        genre_names = []
                        for g in genres:
                            if isinstance(g, dict):
                                genre_names.append(g.get("name", ""))
                            elif isinstance(g, str):
                                genre_names.append(g)
                        book_data["generos"] = ", ".join(filter(None, genre_names))

                # Dados numéricos extras
                if not book_data.get("total_leitores"):
                    book_data["total_leitores"] = book.get("readers_count", 0)
                if not book_data.get("total_resenhas"):
                    book_data["total_resenhas"] = book.get("reviews_count", 0)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Erro ao parsear __NEXT_DATA__: {e}")

    def scrape_range(self, start_id: int, end_id: int, output_file: str = None):
        """
        Coleta dados de livros em um intervalo de IDs, usando threads concorrentes.

        Args:
            start_id: ID inicial
            end_id: ID final (inclusivo)
            output_file: Caminho do arquivo CSV de saída (opcional)
        """
        if output_file is None:
            output_file = config.RAW_BOOKS_FILE

        total_ids = end_id - start_id + 1
        workers = config.MAX_WORKERS

        logger.info(
            f"Iniciando coleta concorrente: IDs {start_id}-{end_id} "
            f"({total_ids} IDs, {workers} workers)"
        )

        book_ids = list(range(start_id, end_id + 1))
        progress = tqdm(total=total_ids, desc="Coletando livros")
        found_count = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submeter todos os IDs como futures
            futures = {
                executor.submit(_scrape_single_book_standalone, bid): bid
                for bid in book_ids
            }

            for future in as_completed(futures):
                book_id = futures[future]
                try:
                    book = future.result()
                    if book is not None:
                        with _lock:
                            self.scraped_books.append(book)
                            found_count += 1

                            # Salvar periodicamente (a cada 100 livros)
                            if found_count % 100 == 0:
                                self._save_to_csv(output_file)
                                logger.info(
                                    f"Progresso: {found_count} livros coletados"
                                )
                except Exception as e:
                    logger.error(f"Erro no ID {book_id}: {e}")

                progress.update(1)

        progress.close()

        # Salvar resultado final
        self._save_to_csv(output_file)
        logger.info(
            f"Coleta finalizada! {len(self.scraped_books)} livros salvos em {output_file}"
        )

        return self.scraped_books

    def _save_to_csv(self, output_file: str):
        """Salva os livros coletados em arquivo CSV."""
        if not self.scraped_books:
            return

        # Coletar todos os campos possíveis
        all_fields = set()
        for book in self.scraped_books:
            all_fields.update(book.keys())
        fieldnames = sorted(all_fields)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.scraped_books)

    def close(self):
        """Fecha o cliente HTTP."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _scrape_single_book_standalone(book_id: int) -> dict | None:
    """
    Função standalone para scraping de um livro (usada em threads).
    Cada chamada cria seu próprio cliente HTTP para thread safety.
    """
    import time, random

    # Thread-local client para evitar conflitos
    client = SkoobClient()
    url = BookScraper.BOOK_URL_TEMPLATE.format(book_id=book_id)

    try:
        html = client.get_html(url)
        if html is None:
            return None

        if "/login" in html[:500]:
            return None

        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("title")
        if title_tag and title_tag.string and title_tag.string.strip() == "Skoob":
            return None

        book_data = {"book_id": book_id, "url": url}

        # JSON-LD extraction
        ld_script = soup.find("script", type="application/ld+json")
        if ld_script and ld_script.string:
            try:
                ld_data = json.loads(ld_script.string)
                if ld_data.get("@type") == "Book":
                    book_data["titulo"] = ld_data.get("name", "")

                    author = ld_data.get("author", {})
                    if isinstance(author, dict):
                        book_data["autor"] = author.get("name", "")
                    elif isinstance(author, list) and len(author) > 0:
                        book_data["autor"] = author[0].get("name", "")

                    book_data["imagem_capa"] = ld_data.get("image", "")
                    book_data["sinopse"] = ld_data.get("description", "")

                    rating = ld_data.get("aggregateRating", {})
                    if rating:
                        book_data["nota_media"] = float(rating.get("ratingValue", 0))
                        book_data["total_avaliacoes"] = int(rating.get("ratingCount", 0))

                    book_data["isbn"] = ld_data.get("isbn", "")

                    publisher = ld_data.get("publisher", {})
                    if isinstance(publisher, dict):
                        book_data["editora"] = publisher.get("name", "")

                    book_data["num_paginas"] = ld_data.get("numberOfPages", "")
                    book_data["data_publicacao"] = ld_data.get("datePublished", "")

                    genre = ld_data.get("genre", "")
                    if genre:
                        if isinstance(genre, list):
                            book_data["generos"] = ", ".join(genre)
                        else:
                            book_data["generos"] = genre
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # HTML fallbacks
        if not book_data.get("titulo"):
            h1 = soup.find("h1")
            if h1:
                book_data["titulo"] = h1.get_text(strip=True)

        # Reader stats
        filter_mapping = {
            "filter=read": "leitores_leram",
            "filter=reading": "leitores_lendo",
            "filter=want_to_read": "leitores_quero_ler",
            "filter=rereading": "leitores_relendo",
            "filter=abandoned": "leitores_abandonaram",
        }
        for filter_key, field_name in filter_mapping.items():
            link = soup.find("a", href=re.compile(re.escape(filter_key)))
            if link:
                for span in link.find_all("span"):
                    text = span.get_text(strip=True).replace(".", "").replace(",", "")
                    if text.isdigit():
                        book_data[field_name] = int(text)
                        break

        total_leitores = sum(book_data.get(f, 0) for f in filter_mapping.values())
        if total_leitores > 0:
            book_data["total_leitores"] = total_leitores

        # Reviews count
        review_link = soup.find("a", href=re.compile(r"/reviews"))
        if review_link:
            for span in review_link.find_all("span"):
                text = span.get_text(strip=True).replace(".", "").replace(",", "")
                if text.isdigit():
                    book_data["total_resenhas"] = int(text)
                    break

        if not book_data.get("generos"):
            # Fallback 1: Next.js Pages Router (__NEXT_DATA__)
            nd_script = soup.find("script", id="__NEXT_DATA__")
            if nd_script and getattr(nd_script, "string", None):
                try:
                    d = json.loads(nd_script.string)
                    b_data = d.get("props", {}).get("pageProps", {}).get("book", {})
                    g_list = b_data.get("genres", b_data.get("tags", []))
                    if isinstance(g_list, list):
                        g_names = [g.get("name", g) if isinstance(g, dict) else g for g in g_list]
                        g_names = list(dict.fromkeys(filter(None, g_names)))
                        if g_names:
                            book_data["generos"] = ", ".join(g_names)
                except Exception:
                    pass

            # Fallback 2: Next.js App Router (Flight data) regex
            if not book_data.get("generos"):
                # O HTML pode conter strings JSON escapadas (ex: \"genres\")
                unescaped_html = html.replace('\\"', '"')
                genres_match = re.search(r'"genres"\s*:\s*\{"title"\s*:\s*"Gêneros"\s*,\s*"items"\s*:\s*(\[.*?\])\}', unescaped_html)
                if genres_match:
                    try:
                        g_items = json.loads(genres_match.group(1))
                        g_names = [item.get("name") for item in g_items if isinstance(item, dict) and item.get("name")]
                        # Deduplicate while preserving order
                        g_names = list(dict.fromkeys(g_names))
                        if g_names:
                            book_data["generos"] = ", ".join(g_names)
                    except Exception:
                        pass
        
            # Fallback 3: Genre tags a hrefs
            if not book_data.get("generos"):
                tag_els = soup.find_all("a", href=re.compile(r"/tag/|/genre/|/category/"))
                if tag_els:
                    tags = [t.get_text(strip=True) for t in tag_els if t.get_text(strip=True)]
                    if tags:
                        tags = list(dict.fromkeys(tags))
                        book_data["generos"] = ", ".join(tags)

        if not book_data.get("titulo"):
            return None

        return book_data

    except Exception as e:
        logger.debug(f"Erro ao scraping livro {book_id}: {e}")
        return None
    finally:
        client.close()


# ============================================================
# Execução direta para teste
# ============================================================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  BookScraper - Teste de Coleta Concorrente")
    print("=" * 60)

    # Teste rápido: 20 IDs com scraping concorrente
    with BookScraper() as scraper:
        books = scraper.scrape_range(1, 20)
        print(f"\nLivros encontrados: {len(books)}")
        for b in books:
            titulo = b.get("titulo", "?")
            autor = b.get("autor", "?")
            nota = b.get("nota_media", "?")
            print(f"  ID={b['book_id']}: {titulo} - {autor} (nota={nota})")

    print("\nTeste concluído!")

