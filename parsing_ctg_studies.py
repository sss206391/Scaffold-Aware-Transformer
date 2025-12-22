"""
대용량 ctg-studies.json 파일 파싱 스크립트
- 55만개 이상의 임상시험 데이터를 스트리밍 방식으로 처리
- Study Overview, Participation Criteria, Study Plan 섹션 추출
- 메모리 효율적인 처리를 위해 ijson 사용

출력 구조:
{
    "NCT04882579": {
        "Study Overview": {...},
        "Participation Criteria": {...},
        "Study Plan": {...}
    },
    "NCT01275742": {
        "Study Overview": {...},
        "Participation Criteria": {...},
        "Study Plan": {...}
    },
    ...
}
"""

import json
import ijson
import sys
import os
from typing import Dict, Any, Generator
from datetime import datetime


def parse_study_overview(protocol: Dict) -> Dict[str, Any]:
    """Study Overview 섹션 파싱"""
    identification = protocol.get('identificationModule', {})
    description = protocol.get('descriptionModule', {})
    status = protocol.get('statusModule', {})
    design = protocol.get('designModule', {})
    conditions = protocol.get('conditionsModule', {})
    interventions_module = protocol.get('armsInterventionsModule', {})
    sponsor = protocol.get('sponsorCollaboratorsModule', {})
    
    overview = {
        'NCT ID': identification.get('nctId', 'N/A'),
        'Organization Study ID': identification.get('orgStudyIdInfo', {}).get('id', 'N/A'),
        'Brief Title': identification.get('briefTitle', 'N/A'),
        'Official Title': identification.get('officialTitle', 'N/A'),
        'Brief Summary': description.get('briefSummary', 'N/A'),
        'Detailed Description': description.get('detailedDescription', 'N/A'),
        'Conditions': conditions.get('conditions', []),
        'Study Type': design.get('studyType', 'N/A'),
        'Phases': design.get('phases', []),
        'Enrollment': design.get('enrollmentInfo', {}).get('count', 'N/A'),
        'Enrollment Type': design.get('enrollmentInfo', {}).get('type', 'N/A'),
        'Overall Status': status.get('overallStatus', 'N/A'),
        'Start Date': status.get('startDateStruct', {}).get('date', 'N/A'),
        'Primary Completion Date': status.get('primaryCompletionDateStruct', {}).get('date', 'N/A'),
        'Completion Date': status.get('completionDateStruct', {}).get('date', 'N/A'),
        'Lead Sponsor': sponsor.get('leadSponsor', {}).get('name', 'N/A'),
        'Responsible Party': sponsor.get('responsibleParty', {}),
        'Interventions': []
    }
    
    # Interventions 파싱
    for intervention in interventions_module.get('interventions', []):
        overview['Interventions'].append({
            'Type': intervention.get('type', 'N/A'),
            'Name': intervention.get('name', 'N/A'),
            'Description': intervention.get('description', 'N/A'),
            'Arm Group Labels': intervention.get('armGroupLabels', [])
        })
    
    return overview


def parse_participation_criteria(protocol: Dict) -> Dict[str, Any]:
    """Participation Criteria 섹션 파싱"""
    eligibility = protocol.get('eligibilityModule', {})
    
    criteria = {
        'Eligibility Criteria': eligibility.get('eligibilityCriteria', 'N/A'),
        'Healthy Volunteers': 'Yes' if eligibility.get('healthyVolunteers', False) else 'No',
        'Sex': eligibility.get('sex', 'N/A'),
        'Minimum Age': eligibility.get('minimumAge', 'N/A'),
        'Maximum Age': eligibility.get('maximumAge', 'N/A'),
        'Standard Ages': eligibility.get('stdAges', [])
    }
    
    return criteria


def parse_study_plan(protocol: Dict) -> Dict[str, Any]:
    """Study Plan 섹션 파싱"""
    design = protocol.get('designModule', {})
    design_info = design.get('designInfo', {})
    arms_module = protocol.get('armsInterventionsModule', {})
    outcomes = protocol.get('outcomesModule', {})
    
    plan = {
        'Design Information': {
            'Allocation': design_info.get('allocation', 'N/A'),
            'Intervention Model': design_info.get('interventionModel', 'N/A'),
            'Intervention Model Description': design_info.get('interventionModelDescription', 'N/A'),
            'Primary Purpose': design_info.get('primaryPurpose', 'N/A'),
            'Masking': design_info.get('maskingInfo', {}).get('masking', 'N/A'),
            'Masking Description': design_info.get('maskingInfo', {}).get('maskingDescription', 'N/A'),
            'Who Masked': design_info.get('maskingInfo', {}).get('whoMasked', [])
        },
        'Arm Groups': [],
        'Primary Outcomes': [],
        'Secondary Outcomes': [],
        'Other Outcomes': []
    }
    
    # Arm Groups 파싱
    for arm in arms_module.get('armGroups', []):
        plan['Arm Groups'].append({
            'Label': arm.get('label', 'N/A'),
            'Type': arm.get('type', 'N/A'),
            'Description': arm.get('description', 'N/A'),
            'Intervention Names': arm.get('interventionNames', [])
        })
    
    # Primary Outcomes 파싱
    for outcome in outcomes.get('primaryOutcomes', []):
        plan['Primary Outcomes'].append({
            'Measure': outcome.get('measure', 'N/A'),
            'Description': outcome.get('description', 'N/A'),
            'Time Frame': outcome.get('timeFrame', 'N/A')
        })
    
    # Secondary Outcomes 파싱
    for outcome in outcomes.get('secondaryOutcomes', []):
        plan['Secondary Outcomes'].append({
            'Measure': outcome.get('measure', 'N/A'),
            'Description': outcome.get('description', 'N/A'),
            'Time Frame': outcome.get('timeFrame', 'N/A')
        })
    
    # Other Outcomes 파싱
    for outcome in outcomes.get('otherOutcomes', []):
        plan['Other Outcomes'].append({
            'Measure': outcome.get('measure', 'N/A'),
            'Description': outcome.get('description', 'N/A'),
            'Time Frame': outcome.get('timeFrame', 'N/A')
        })
    
    return plan


def parse_single_study(study_data: Dict) -> tuple:
    """단일 임상시험 데이터 파싱"""
    protocol = study_data.get('protocolSection', {})
    
    # NCT ID 추출
    nct_id = protocol.get('identificationModule', {}).get('nctId', 'UNKNOWN')
    
    # 세 섹션 파싱
    parsed = {
        'Study Overview': parse_study_overview(protocol),
        'Participation Criteria': parse_participation_criteria(protocol),
        'Study Plan': parse_study_plan(protocol)
    }
    
    return nct_id, parsed


def stream_studies(json_file: str) -> Generator[Dict, None, None]:
    """대용량 JSON 파일에서 스트리밍으로 항목 읽기"""
    with open(json_file, 'rb') as f:
        # ijson으로 배열의 각 항목을 스트리밍
        parser = ijson.items(f, 'item')
        for item in parser:
            yield item


def parse_ctg_studies_streaming(
    input_file: str, 
    output_file: str = 'parsed_all_studies.json',
    progress_interval: int = 10000,
    max_studies: int = None
):
    """
    스트리밍 방식으로 대용량 JSON 파싱 (메모리 효율적)
    
    Args:
        input_file: 입력 JSON 파일 경로 (ctg-studies.json)
        output_file: 출력 JSON 파일 경로
        progress_interval: 진행 상황 출력 간격
        max_studies: 최대 처리할 항목 수 (None이면 전체 처리)
    """
    print(f"\n{'='*70}")
    print(f"📂 임상시험 데이터 파싱 시작")
    print(f"{'='*70}")
    print(f"입력 파일: {input_file}")
    print(f"출력 파일: {output_file}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 결과 저장용 딕셔너리
    all_studies = {}
    processed = 0
    errors = 0
    
    try:
        for study in stream_studies(input_file):
            try:
                nct_id, parsed_data = parse_single_study(study)
                all_studies[nct_id] = parsed_data
                processed += 1
                
                # 진행 상황 출력
                if processed % progress_interval == 0:
                    print(f"  ✓ {processed:,}개 처리 완료... (최근: {nct_id})")
                
                # 최대 처리 수 제한
                if max_studies and processed >= max_studies:
                    print(f"\n⚠️ 최대 처리 수 ({max_studies:,})에 도달하여 중단합니다.")
                    break
                    
            except Exception as e:
                errors += 1
                if errors <= 10:  # 처음 10개 에러만 출력
                    print(f"  ⚠️ 파싱 오류 (항목 {processed + 1}): {str(e)[:50]}")
                continue
                
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 사용자에 의해 중단됨. 현재까지 처리된 {processed:,}개 저장 중...")
    
    # 결과 저장
    print(f"\n{'='*70}")
    print(f"💾 결과 저장 중... ({len(all_studies):,}개 항목)")
    print(f"{'='*70}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_studies, f, ensure_ascii=False, indent=2)
    
    # 완료 메시지
    file_size = os.path.getsize(output_file)
    print(f"\n{'='*70}")
    print(f"✅ 파싱 완료!")
    print(f"{'='*70}")
    print(f"  • 처리된 항목: {processed:,}개")
    print(f"  • 저장된 항목: {len(all_studies):,}개")
    print(f"  • 오류 발생: {errors:,}개")
    print(f"  • 출력 파일: {output_file}")
    print(f"  • 파일 크기: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
    print(f"  • 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return all_studies


def parse_ctg_studies_chunked(
    input_file: str,
    output_dir: str = 'parsed_chunks',
    chunk_size: int = 50000,
    progress_interval: int = 10000
):
    """
    청크 단위로 저장하는 방식 (메모리 부족 시 사용)
    50,000개씩 나눠서 여러 파일로 저장 후 나중에 병합
    
    Args:
        input_file: 입력 JSON 파일 경로
        output_dir: 출력 디렉토리 경로
        chunk_size: 청크당 항목 수
        progress_interval: 진행 상황 출력 간격
    """
    print(f"\n{'='*70}")
    print(f"📂 임상시험 데이터 청크 파싱 시작")
    print(f"{'='*70}")
    print(f"입력 파일: {input_file}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"청크 크기: {chunk_size:,}개")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    current_chunk = {}
    chunk_num = 1
    processed = 0
    errors = 0
    saved_files = []
    
    try:
        for study in stream_studies(input_file):
            try:
                nct_id, parsed_data = parse_single_study(study)
                current_chunk[nct_id] = parsed_data
                processed += 1
                
                # 진행 상황 출력
                if processed % progress_interval == 0:
                    print(f"  ✓ {processed:,}개 처리 완료... (최근: {nct_id})")
                
                # 청크 저장
                if len(current_chunk) >= chunk_size:
                    chunk_file = os.path.join(output_dir, f'chunk_{chunk_num:04d}.json')
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        json.dump(current_chunk, f, ensure_ascii=False)
                    
                    saved_files.append(chunk_file)
                    print(f"  💾 청크 {chunk_num} 저장 완료: {chunk_file} ({len(current_chunk):,}개)")
                    
                    current_chunk = {}
                    chunk_num += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  ⚠️ 파싱 오류 (항목 {processed + 1}): {str(e)[:50]}")
                continue
                
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 사용자에 의해 중단됨.")
    
    # 마지막 청크 저장
    if current_chunk:
        chunk_file = os.path.join(output_dir, f'chunk_{chunk_num:04d}.json')
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(current_chunk, f, ensure_ascii=False)
        saved_files.append(chunk_file)
        print(f"  💾 청크 {chunk_num} 저장 완료: {chunk_file} ({len(current_chunk):,}개)")
    
    # 완료 메시지
    print(f"\n{'='*70}")
    print(f"✅ 청크 파싱 완료!")
    print(f"{'='*70}")
    print(f"  • 처리된 항목: {processed:,}개")
    print(f"  • 생성된 청크: {len(saved_files)}개")
    print(f"  • 오류 발생: {errors:,}개")
    print(f"  • 출력 디렉토리: {output_dir}")
    print(f"  • 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return saved_files


def merge_chunks(chunk_dir: str, output_file: str = 'parsed_all_studies.json'):
    """청크 파일들을 하나로 병합"""
    print(f"\n{'='*70}")
    print(f"🔗 청크 파일 병합 시작")
    print(f"{'='*70}")
    
    all_studies = {}
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith('.json')])
    
    for i, chunk_file in enumerate(chunk_files, 1):
        file_path = os.path.join(chunk_dir, chunk_file)
        print(f"  읽는 중: {chunk_file} ({i}/{len(chunk_files)})")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            all_studies.update(chunk_data)
    
    print(f"\n💾 병합 결과 저장 중... ({len(all_studies):,}개 항목)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_studies, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_file)
    print(f"\n✅ 병합 완료!")
    print(f"  • 총 항목: {len(all_studies):,}개")
    print(f"  • 출력 파일: {output_file}")
    print(f"  • 파일 크기: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
    
    return all_studies


# 메인 실행
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='대용량 임상시험 JSON 파싱')
    parser.add_argument('input_file', nargs='?', default='ctg-studies.json',
                        help='입력 JSON 파일 경로 (기본값: ctg-studies.json)')
    parser.add_argument('-o', '--output', default='parsed_all_studies.json',
                        help='출력 JSON 파일 경로 (기본값: parsed_all_studies.json)')
    parser.add_argument('-m', '--mode', choices=['single', 'chunked'], default='single',
                        help='파싱 모드: single(단일 파일) 또는 chunked(청크 분할)')
    parser.add_argument('-c', '--chunk-size', type=int, default=50000,
                        help='청크 모드에서 청크당 항목 수 (기본값: 50000)')
    parser.add_argument('-n', '--max-studies', type=int, default=None,
                        help='최대 처리할 항목 수 (테스트용)')
    parser.add_argument('--merge', action='store_true',
                        help='청크 파일들을 병합')
    parser.add_argument('--chunk-dir', default='parsed_chunks',
                        help='청크 디렉토리 경로 (기본값: parsed_chunks)')
    
    args = parser.parse_args()
    
    # 병합 모드
    if args.merge:
        merge_chunks(args.chunk_dir, args.output)
        sys.exit(0)
    
    # 파일 존재 확인
    if not os.path.exists(args.input_file):
        print(f"❌ 파일을 찾을 수 없습니다: {args.input_file}")
        sys.exit(1)
    
    # 파싱 실행
    if args.mode == 'single':
        # 단일 파일로 저장 (메모리가 충분한 경우)
        parse_ctg_studies_streaming(
            args.input_file,
            args.output,
            max_studies=args.max_studies
        )
    else:
        # 청크로 나눠서 저장 (메모리가 부족한 경우)
        parse_ctg_studies_chunked(
            args.input_file,
            args.chunk_dir,
            args.chunk_size
        )
        
        # 청크 병합 여부 확인
        response = input("\n청크 파일들을 하나로 병합하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            merge_chunks(args.chunk_dir, args.output)