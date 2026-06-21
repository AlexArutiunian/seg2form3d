# Security

## Секреты

В Git нельзя добавлять `.env`, приватные ключи, GitHub PAT, адреса лабораторной сети, SSH usernames/hosts, robot IP, datasets или checkpoints. Публичный `.env.example` содержит только loopback и пустые placeholders.

`.env` не шифрует значения. Он только отделяет machine-local config от публичного кода. Пароли и токены должны храниться в OS credential store; SSH-доступ — в отдельном ключе с passphrase и минимальными правами.

GitHub-аутентификацию выполняйте через `gh auth login --web` или SSH-ключ с passphrase. Не вставляйте PAT в remote URL и не сохраняйте его в `.env`.

## Сеть

- inference-серверы bind только на `127.0.0.1`;
- клиент подключается через SSH local forwarding;
- SSH использует key auth, `BatchMode=yes` и `ExitOnForwardFailure=yes`;
- robot camera доступна только из лабораторной LAN/VPN;
- firewall сервера не должен публиковать segmentation ports.

Legacy segmentation protocol использует `pickle` для совместимости. Unpickle данных от недоверенного peer может выполнить код, поэтому endpoint нельзя выставлять наружу или проксировать для третьих лиц. План миграции: versioned JSON/msgpack envelope и PNG/RLE binary payload без pickle.

## Если секрет уже попал в историю

1. Сразу отозвать/заменить пароль, PAT или SSH key. Очистка Git не отменяет уже скопированный секрет.
2. Переписать всю публичную историю (`git filter-repo` либо новый root commit).
3. Force-push всех веток и тегов.
4. Проверить GitHub Actions logs, releases, forks и caches.

Ранние commits этого репозитория содержали инфраструктурный SSH host/user и внутренний IP робота. Их следует считать раскрытыми; после подготовки безопасного snapshot история должна быть заменена, а соответствующие реквизиты ротированы.
