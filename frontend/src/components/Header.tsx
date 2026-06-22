export function Header() {
  return (
    <>
      <header className="site-header">
        <h1>Climate Signal</h1>
        <p>GHG Emissions Classification System — Country-Year Analysis Platform</p>
      </header>
      <div className="site-subnav">
        Data Source: Our World in Data CO₂ Emissions Dataset &nbsp;|&nbsp;
        Coverage: 161 Jurisdictions &nbsp;|&nbsp;
        Methodology: Supervised Machine Learning Classification
      </div>
    </>
  );
}