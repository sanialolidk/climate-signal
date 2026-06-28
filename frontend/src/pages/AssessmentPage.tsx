import { CountryAssessment } from "../components/CountryAssessment";
import type { Meta } from "../types";

interface Props {
  meta: Meta;
}

export function AssessmentPage({ meta }: Props) {
  return <CountryAssessment meta={meta} />;
}