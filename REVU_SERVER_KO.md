# LUPER REVU 수집 서버

LUPER의 기존 Render 웹 서비스 안에서 동작하는 독립 수집 모듈입니다.

## 현재 검증 범위

- 상태 확인: `GET /api/revu/health`
- 접속 키 확인: `GET /api/revu/auth-check`
- 배민 수집: `POST /api/revu/collect/baemin`
- 요청 중 전달된 로그인 헤더와 수집 결과는 DB·디스크에 저장하지 않습니다.
- 요청 대상은 `https://self-api.baemin.com/v1/review/shops/.../reviews`로 제한됩니다.
- 외부 주소로의 리다이렉트는 차단합니다.

## 인증

`X-Revu-Key` 헤더에는 Render의 `REVU_COLLECTOR_KEY` 값을 사용합니다. 이 환경 변수가
없으면 기존 `ADMIN_KEY`를 사용합니다. 키가 설정되지 않았거나 기본값이면 수집 요청을
거부합니다.

## 배민 요청 예시

```json
{
  "url": "https://self-api.baemin.com/v1/review/shops/매장번호/reviews?from=2026-06-01&to=2026-09-22",
  "headers": {
    "authorization": "브라우저에서 캡처한 임시 인증값"
  },
  "pageSize": 10,
  "maxPages": 100,
  "timeoutSeconds": 25
}
```

응답에는 `count`, `pages`, `elapsedMs`, 페이지별 `responseMs`, `reviews`가 포함됩니다.
이번 단계에서는 속도와 연결을 검증하기 위해 동기 방식으로 실행하며 Gunicorn 요청 제한은
120초입니다.
