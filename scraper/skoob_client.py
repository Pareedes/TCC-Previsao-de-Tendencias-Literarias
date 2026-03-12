"""
Cliente HTTP base para comunicação com o Skoob.
Gerencia sessão, rate limiting, retries e logging.
"""

import time
import random
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SkoobClient")


class SkoobClient:
    """
    Cliente HTTP robusto para fazer requisições ao Skoob.

    Funcionalidades:
    - Session persistente (reutiliza conexões TCP)
    - Headers realistas para evitar bloqueios
    - Rate limiting automático entre requisições
    - Retry com backoff exponencial em caso de falha
    - Logging detalhado de cada requisição
    """

    def __init__(self):
        self.session = self._create_session()
        self._last_request_time = 0
        self._request_count = 0

    def _create_session(self) -> requests.Session:
        """Cria uma sessão HTTP com retry automático."""
        session = requests.Session()
        session.headers.update(config.HTTP_HEADERS)

        # Configurar retry automático para erros de rede
        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            backoff_factor=config.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _rate_limit(self):
        """Aplica delay entre requisições para respeitar o servidor."""
        elapsed = time.time() - self._last_request_time
        delay = random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX)
        if elapsed < delay:
            wait_time = delay - elapsed
            logger.debug(f"Rate limit: aguardando {wait_time:.1f}s")
            time.sleep(wait_time)

    def get(self, url: str, params: dict = None) -> requests.Response | None:
        """
        Faz uma requisição GET com rate limiting e tratamento de erros.

        Args:
            url: URL a ser acessada
            params: Parâmetros de query string opcionais

        Returns:
            Response do requests, ou None se todas as tentativas falharem
        """
        self._rate_limit()
        self._request_count += 1

        try:
            logger.info(
                f"[#{self._request_count}] GET {url}"
                + (f" params={params}" if params else "")
            )
            response = self.session.get(
                url, params=params, timeout=config.REQUEST_TIMEOUT
            )
            self._last_request_time = time.time()

            if response.status_code == 200:
                logger.debug(f"  → OK ({len(response.content)} bytes)")
                return response
            elif response.status_code == 404:
                logger.warning(f"  → 404 Not Found: {url}")
                return None
            else:
                logger.warning(f"  → HTTP {response.status_code}: {url}")
                response.raise_for_status()

        except requests.exceptions.Timeout:
            logger.error(f"  → Timeout após {config.REQUEST_TIMEOUT}s: {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"  → Erro de conexão: {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"  → Erro HTTP: {e}")
            return None
        except Exception as e:
            logger.error(f"  → Erro inesperado: {e}")
            return None

        return None

    def get_html(self, url: str, params: dict = None) -> str | None:
        """
        Faz GET e retorna o conteúdo HTML como string.

        Returns:
            String HTML do corpo da resposta, ou None em caso de erro
        """
        response = self.get(url, params)
        if response is not None:
            response.encoding = "utf-8"
            return response.text
        return None

    def get_json(self, url: str, params: dict = None) -> dict | None:
        """
        Faz GET e retorna o conteúdo JSON como dicionário.

        Returns:
            Dicionário com os dados JSON, ou None em caso de erro
        """
        response = self.get(url, params)
        if response is not None:
            try:
                return response.json()
            except ValueError:
                logger.error(f"  → Resposta não é JSON válido: {url}")
                return None
        return None

    @property
    def total_requests(self) -> int:
        """Retorna o total de requisições feitas."""
        return self._request_count

    def close(self):
        """Fecha a sessão HTTP."""
        self.session.close()
        logger.info(f"Sessão encerrada. Total de requisições: {self._request_count}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
