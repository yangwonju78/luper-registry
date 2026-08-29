LUPER SERVER DB 저장소

이 폴더는 앱/서버 코드 버전과 분리해서 유지합니다.

구조
DB/registry.db                  로컬 SQLite 메타데이터(Managed PostgreSQL 사용 시 생성되지 않을 수 있음)
DB/images/original/             등록 원본 이미지
DB/images/derived/              재분석으로 다시 생성 가능한 ROI 이미지

중요
1. 새 코드 버전을 배포할 때 이 DB 폴더를 덮어쓰거나 삭제하지 마세요.
2. 기존 서버가 BLOB 방식이면 V0.8.2.5 최초 실행 시 원본 이미지를 images/original로 자동 이관합니다.
3. source_sha256 기반 파일명이라 같은 이미지 중복 저장을 줄입니다.
4. Render 같은 호스팅에서 서버 파일시스템이 임시(ephemeral)이면 DB 폴더에 Persistent Disk를 연결해야 합니다.
5. DATABASE_URL이 있으면 등록 메타데이터는 PostgreSQL에 유지되고, 이미지 파일만 이 DB 폴더를 사용합니다.
